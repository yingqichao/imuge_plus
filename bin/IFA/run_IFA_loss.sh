python -m torch.distributed.launch --master_port 5555 --nproc_per_node=1 train.py \
                                -opt options/train/IFA/train_IFA_baseline.yml -mode 3 -task_name PSNR_predict_resnet \
                                --launcher pytorch
