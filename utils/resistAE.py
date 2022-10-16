import os
import numpy as np
import math
import torch
import torchvision
import imageio
import foolbox
from torch import nn


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# normalization
mean = (torch.FloatTensor([125.3, 123.0, 113.9]) / 255.0).cuda()
std = (torch.FloatTensor([63.0, 62.1, 66.7]) / 255.0).cuda()
subtrans = torchvision.transforms.Normalize(mean,std)

def my_trans(pic):
    return subtrans(pic/255.)

globe_Qy=torch.Tensor([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
globe_Qc=torch.Tensor([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])
globe_w = torch.Tensor([
    [0.299,0.587,0.114],
    [-0.168736,-0.331264,0.5],
    [0.5,-0.418688,-0.081312]
    ])
globe_b = torch.Tensor([
    [0],
    [128],
    [128]
    ])

def resistGetOutput(image,my_model,q):
    image_x,image_y,image_z = image.shape
    mypic = image
    my_input = mypic.cuda()

    x = my_input.reshape(3,-1)
    x = x.cuda()
    w = globe_w.cuda()
    b = globe_b.cuda()
    YUV = torch.matmul(w,x) + b

    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i==0:
                a = torch.tensor(1/8).sqrt()
            else:
                a = torch.tensor(1/4).sqrt()
            A[i,j] = a * math.cos(math.pi*(j+0.5)*i/8)
    YUV = YUV.reshape(image_x,image_y,image_z)
    A = torch.Tensor(A)
    A = A.cuda()

    q_dct = torch.FloatTensor(image_x,image_y,image_z).fill_(0)
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    q_dct = q_dct.cuda()
    Qy = (globe_Qy*s).cuda()
    Qc = (globe_Qc*s).cuda()
    
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y*image_z//64,1)
        else:
            Q_mat = Qc.repeat(image_y*image_z//64,1)
        a = torch.cat(YUV[i].split(8,0),1)
        a = torch.matmul(A,a)
        a = torch.cat(a.split(8,1),0)
        a = torch.div(torch.matmul(a,A.t()),Q_mat)
        a = torch.cat(a.split(8,0),1)
        q_dct[i] = torch.cat(a.split(image_z,1),0)

    quantanized_dct = q_dct.round() + (q_dct - q_dct.round())**3

    i_YUV = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        a = quantanized_dct[i] * Q_mat
        a = torch.cat(a.split(8,0),1)
        a = torch.matmul(A.t(),a)
        a = torch.cat(a.split(8,1),0)
        a = torch.matmul(a,A)
        a = torch.cat(a.split(8,0),1)
        i_YUV[i] = torch.cat(a.split(image_z,1),0)

    i_YUV = i_YUV.reshape(3,-1)
    i_x = torch.matmul(w.inverse(),i_YUV - b)
    real_input = i_x.reshape(image_x,image_y,image_z)

    t1 = my_model(my_trans(real_input).cuda().unsqueeze(0))
    my_output = t1.reshape(1,-1)
    return my_output

# 根据模型对图像进行预测
def myPredict(model,image):
    return model(my_trans(image.cuda().permute(2,0,1)).unsqueeze(0))

class my_network(nn.Module):
    def __init__(self,model,q):
        super(my_network,self).__init__()
        self.model = model
        self.q = q
    def forward(self,input):
        return resistGetOutput(input[0],self.model,self.q)

def PGD_resist(model,image,target,q):
    perturbed = image.cuda().clone().permute(2,0,1)
    fmodel = foolbox.PyTorchModel(my_network(model,q).cuda().eval(),bounds=(0,255))
    attack = foolbox.attacks.LinfPGD(steps=10,rel_stepsize=0.1,random_start=False)
    criterion = foolbox.Misclassification(torch.LongTensor([target]).cuda())
    a,perturbed,success = attack(fmodel,perturbed.unsqueeze(0),criterion,epsilons=3)
    perturbed = perturbed[0]
    return perturbed.permute(1,2,0)

if __name__ == '__main__':
    # 导入网络，此处使用了pytorch中预训练的resnet50
    my_model = torchvision.models.resnet50(pretrained=True).cuda().eval()
    # 导入原始图像和类别标签,示例图像来自于ImageNet验证集
    image = imageio.imread('3.png')
    image = torch.FloatTensor(np.asarray(image)).cuda()
    label = 2
    print("正确标签为：",label)
    # 得到预测结果
    predict_label = torch.argmax(myPredict(my_model,image),1).item()
    print("预测标签为：",predict_label)
    label = 2
    # launch resist attack
    perturbed = PGD_resist(my_model,image,label,90)
    adv_label = torch.argmax(myPredict(my_model,perturbed),1).item()
    print("对抗样本标签为：",adv_label)
    # 输出平均L2范数
    print(torch.sqrt(torch.mean((perturbed-image)** 2)).item())