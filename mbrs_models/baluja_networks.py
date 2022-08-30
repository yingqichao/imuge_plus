# Preparation Network (2 conv layers)
import torch.nn as nn
import torch

class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out


# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self,n_dim=3):
        super(HidingNetwork, self).__init__()
        self.initialH3 = nn.Sequential(
            nn.Conv2d(n_dim, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.initialH4 = nn.Sequential(
            nn.Conv2d(n_dim, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.initialH5 = nn.Sequential(
            nn.Conv2d(n_dim, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0),
            # nn.Tanh()
        )

    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)
        # out_noise = gaussian(out.data, 0, 0.1)
        return (torch.tanh(out) + 1) / 2

class RevealNetwork(nn.Module):
    def __init__(self, in_number=3, out_number=3):
        super(RevealNetwork, self).__init__()
        self.out_number = out_number
        self.initialR3 = nn.Sequential(
            nn.Conv2d(in_number, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.initialR4 = nn.Sequential(
            nn.Conv2d(in_number, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.finalR3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, out_number, kernel_size=1, padding=0),
        )
        self.finalR4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(50, out_number, kernel_size=1, padding=0),
        )
        self.final = nn.Conv2d(out_number*2, out_number, kernel_size=1, padding=0)

        # self.qf_pred_1 = sequential(
        #                           torch.nn.AdaptiveAvgPool2d((1, 1)),
        #                           torch.nn.Flatten(),
        #                           torch.nn.Linear(50, 50),
        #                           nn.ReLU(),
        #                           torch.nn.Linear(50, 50),
        #                           nn.ReLU(),
        #                           torch.nn.Linear(50, 1),
        #                           nn.Sigmoid()
        #                           )
        #
        # self.qf_pred_2 = sequential(
        #                           torch.nn.AdaptiveAvgPool2d((1, 1)),
        #                           torch.nn.Flatten(),
        #                           torch.nn.Linear(50, 50),
        #                           nn.ReLU(),
        #                           torch.nn.Linear(50, 50),
        #                           nn.ReLU(),
        #                           torch.nn.Linear(50, 1),
        #                           nn.Sigmoid()
        #                           )

    def forward(self, r, rjpg=None, double=0.5):
        if rjpg is not None:
            r1 = self.initialR3(r)
            r2 = self.finalR3(r1)
            l1 = self.initialR4(r)
            l2 = self.finalR4(l1)
            # r2 = (torch.tanh(r2) + 1) / 2
            # l2 = (torch.tanh(l2) + 1) / 2
            # a1 = self.qf_pred_1(r1).expand(-1,3).expand(-1,3)
            # a2 = self.qf_pred_2(l1).expand(-1,3).expand(-1,3)

            out = double*r2+(1-double)*l2 # if double else 0.9*r2+0.1*l2

            rjpg1 = self.initialR3(rjpg)
            rjpg2 = self.finalR3(rjpg1)
            ljpg1 = self.initialR4(rjpg)
            ljpg2 = self.finalR4(ljpg1)

            out_jpg = self.final(torch.cat((rjpg2,ljpg2),dim=1).clone().detach())
            # l2 = (torch.tanh(l2) + 1) / 2

            return out, out_jpg, r2, l2, rjpg2, ljpg2 #(torch.tanh(out) + 1) / 2
        else:
            rjpg1 = self.initialR3(r)
            rjpg2 = self.finalR3(rjpg1)
            ljpg1 = self.initialR4(r)
            ljpg2 = self.finalR4(ljpg1)

            out_jpg = self.final(torch.cat((rjpg2, ljpg2), dim=1))
            return out_jpg

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# Reveal Network (2 conv layers)
# class RevealNetwork(nn.Module):
#     def __init__(self, in_number=3, out_number=3):
#         super(RevealNetwork, self).__init__()
#         self.out_number = out_number
#         self.initialR3 = nn.Sequential(
#             nn.Conv2d(in_number, 50, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.initialR4 = nn.Sequential(
#             nn.Conv2d(in_number, 50, kernel_size=4, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=4, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.initialR5 = nn.Sequential(
#             nn.Conv2d(in_number, 50, kernel_size=5, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.finalR3 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.finalR4 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=4, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.finalR5 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=5, padding=2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         self.finalR = nn.Sequential(
#             nn.Conv2d(150, out_number, kernel_size=1, padding=0),
#             # nn.Tanh()
#         )
#
#     def forward(self, r):
#         r1 = self.initialR3(r)
#         r2 = self.initialR4(r)
#         r3 = self.initialR5(r)
#         mid = torch.cat((r1, r2, r3), 1)
#         r4 = self.finalR3(mid)
#         r5 = self.finalR4(mid)
#         r6 = self.finalR5(mid)
#         mid2 = torch.cat((r4, r5, r6), 1)
#         out = self.finalR(mid2)
#
#         return (torch.tanh(out) + 1) / 2



# Join three networks in one module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2, x_2_noise = self.m2(mid)
        x_3 = self.m3(x_2_noise)
        return x_2, x_3