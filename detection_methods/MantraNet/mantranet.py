import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
#Pytorch
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from noise_layers.combined import Combined
from noise_layers import *
from noise_layers.resize import Resize
from noise_layers.gaussian_blur import GaussianBlur
#pytorch-lightning
# import pytorch_lightning as pl

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##reproduction of the hardsigmoid coded in tensorflow (which is not exactly the same one in Pytorch)
def hardsigmoid(T):
    T_0 = T
    T = 0.2 * T_0 + 0.5
    T[T_0 < -2.5] = 0
    T[T_0 > 2.5] = 1

    return T

##ConvLSTM - Equivalent implementation of ConvLSTM2d in pytorch
##Source : https://github.com/ndrplz/ConvLSTM_pytorch
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.sigmoid = hardsigmoid

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        c_next = f * c_cur + i * torch.tanh(cc_c)
        o = self.sigmoid(cc_o)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.transpose(0, 1)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvGruCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvGRU cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGruCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.sigmoid = hardsigmoid

        self.conv1 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
                              
        self.conv2 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias)

    
    def forward(self, input_tensor, cur_state):
        h_cur = cur_state

        # print(h_cur)
        h_x = torch.cat([h_cur,input_tensor], dim=1)  # concatenate along channel axis
        
        # print('OK')
        combined_conv = self.conv1(h_x)
        cc_r, cc_u = torch.split(combined_conv, self.hidden_dim, dim=1)
        r = self.sigmoid(cc_r)
        u = self.sigmoid(cc_u)
        
        x_r_o_h=torch.cat([input_tensor,r*h_cur],dim=1)
        # print(x_r_o_h.size())
        combined_conv = self.conv2(x_r_o_h)
        
        c = nn.Tanh()(combined_conv)
        h_next = (1-u)*h_cur+u*c

        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv1.weight.device)


class ConvGRU(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convgru = ConvGRU(64, 16, 3, 1, True, True, False)
        >> _, last_states = convgru(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGruCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.transpose(0, 1)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


## Symmetric padding (not existing natively in Pytorch)
## Source : https://discuss.pytorch.org/t/symmetric-padding/19866/3

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


#batch normalization equivalent to the one proposed in tensorflow
#Source : https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html

def batch_norm(X, eps=0.001):
    # extract the dimensions
    N, C, H, W = X.shape
    device=X.device
    # mini-batch mean
    mean = X.mean(axis=(0, 2, 3)).to(device)
    # mini-batch variance
    variance = ((X - mean.view((1, C, 1, 1))) ** 2).mean(axis=(0, 2, 3)).to(device)
    # normalize
    X = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.pow((variance.view((1, C, 1, 1)) + eps), 0.5)
    return X.to(device)


#MantraNet (equivalent from the one coded in tensorflow at https://github.com/ISICV/ManTraNet)
class MantraNet(nn.Module):
    def __init__(self, in_channel=3, eps=10 ** (-6),device=device):
        super(MantraNet, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()
        self.device=device


        # ********** IMAGE MANIPULATION TRACE FEATURE EXTRACTOR *********

        ## Initialisation

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).to(self.device)
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).to(self.device)
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data=torch.load('MantraNetv4.pt')['SRMConv2D.weight']

        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0)]
        )

        # ********** LOCAL ANOMALY DETECTOR *********

        self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)

        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        self.convlstm = ConvLSTM(input_dim=64,
                                 hidden_dim=8,
                                 kernel_size=(7, 7),
                                 num_layers=1,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.end = nn.Sequential(
            nn.Conv2d(8, 1, 7, 1, padding=3),
            ####################################################################################################
            # todo: Note the sigmoid should be removed if we use bce_with_logit_loss
            ####################################################################################################
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        B, nb_channel, H, W = x.shape
        
        if not(self.training):
            self.GlobalPool = nn.AvgPool2d((H, W), stride=1)
        else:
            if not hasattr(self, 'GlobalPool'):
                self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        # Normalization
        # x = x / 255. * 2 - 1

        ## Image Manipulation Trace Feature Extractor

        ## **Bayar constraints**

        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        #Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        for layer in self.middle_and_last_block:

            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))

            last_block = layer(last_block)

        #L2 normalization
        last_block = F.normalize(last_block, dim=1, p=2)
        

        ## Local Anomaly Feature Extraction
        X_adapt = self.adaptation(last_block)
        X_adapt = batch_norm(X_adapt)

        # Z-pool concatenation
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)
        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)

        mu_31 = self.pool31(X_adapt)
        zpool31 = torch.abs((mu_31 - X_adapt) * inv_sigma_T)

        mu_15 = self.pool15(X_adapt)
        zpool15 = torch.abs((mu_15 - X_adapt) * inv_sigma_T)

        mu_7 = self.pool7(X_adapt)
        zpool7 = torch.abs((mu_7 - X_adapt) * inv_sigma_T)

        input_lstm = torch.cat([zpool7.unsqueeze(0), zpool15.unsqueeze(0), zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0)], axis=0)

        # Conv2DLSTM
        _, output_lstm = self.convlstm(input_lstm)
        output_lstm = output_lstm[0][0]

        final_output = self.end(output_lstm)
        

        return output_lstm, final_output
            

#Slight modification of the original MantraNet using a GRU instead of a LSTM
class MantraNet_GRU(nn.Module):
    def __init__(self,device,in_channel=3,eps=10 **(-4)):
        super(MantraNet_GRU, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()
        self.device=device

        # ********** IMAGE MANIPULATION TRACE FEATURE EXTRACTOR *********

        ## Initialisation

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        
        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
       
        self.SRMConv2D.weight.data=torch.load('MantraNetv4.pt')['SRMConv2D.weight']

        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False
           

        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0)]
        )

        # ********** LOCAL ANOMALY DETECTOR *********

        self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)

        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        self.convgru = ConvGRU(input_dim=64,
                                 hidden_dim=8,
                                 kernel_size=(7, 7),
                                 num_layers=1,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3),nn.Sigmoid())
        
    
        
        self.bayar_mask = torch.ones((5, 5),device=self.device)
        self.bayar_final = torch.zeros((5, 5),device=self.device)
        
    def forward(self, x):
        B, nb_channel, H, W = x.shape
        
        if not(self.training):
            self.GlobalPool = nn.AvgPool2d((H, W), stride=1)
        else:
            if not hasattr(self, 'GlobalPool'):
                self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        # Normalization
        # x = x / 255. * 2 - 1

        ## Image Manipulation Trace Feature Extractor

        ## **Bayar constraints**

        self.bayar_mask[2, 2] = 0
        self.bayar_final[2, 2] = -1

        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final
        
        #Symmetric padding
        X = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(X)
        conv_bayar = self.BayarConv2D(X)
        conv_srm = self.SRMConv2D(X)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        for layer in self.middle_and_last_block:

            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))

            last_block = layer(last_block)

        #L2 normalization
        last_block = F.normalize(last_block, dim=1, p=2)
        
      
        ## Local Anomaly Feature Extraction
        X_adapt = self.adaptation(last_block)
        X_adapt = batch_norm(X_adapt)

        # Z-pool concatenation
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)
        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)

        mu_31 = self.pool31(X_adapt)
        zpool31 = torch.abs((mu_31 - X_adapt) * inv_sigma_T)

        mu_15 = self.pool15(X_adapt)
        zpool15 = torch.abs((mu_15 - X_adapt) * inv_sigma_T)

        mu_7 = self.pool7(X_adapt)
        zpool7 = torch.abs((mu_7 - X_adapt) * inv_sigma_T)

        input_gru = torch.cat([zpool7.unsqueeze(0), zpool15.unsqueeze(0), zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0)], axis=0)

        # Conv2DLSTM
        _,output_gru = self.convgru(input_gru)
        output_gru = output_gru[0]

        final_output = self.end(output_gru)

        return final_output

from PIL import Image
import torchvision.transforms.functional as functional

def image_to_tensor(img):
    img = Image.fromarray(img)
    img_t = functional.to_tensor(np.asarray(img)).float()
    return img_t

##Use pre-trained weights :
def pre_trained_model(weight_path='./MantraNetv4.pt'):
    model=MantraNet()
    model.load_state_dict(torch.load(weight_path))
    return model

import torchvision
#predict a forgery mask of an image
from noise_layers.jpeg import Jpeg
from noise_layers.identity import Identity

jpeg = Combined(
            [JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70), JpegMask(60), Jpeg(60)]).cuda()
resize = Resize().cuda()
gaussian_blur = GaussianBlur().cuda()
identity = Identity().cuda()

def check_forgery(model,fold_path='./',img_path='example.jpg',device=device):


    model.to(device)
    model.eval()
    new_size = 256
    F1_sum, valid = 0, 0
    test_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/tamper_COCO_0114')
    ori_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/ori_COCO_0114')
    mask_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/binary_masks_COCO_0114')

    for idx in range(len(test_data)):
        p, q, r = test_data[idx]
        img_path = os.path.join(p, q, r)
        print(img_path)
        img = cv2.imread(img_path)
        ori_size = img.shape
        img = cv2.resize(img, (new_size, new_size))
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))

        p, q, r = ori_data[idx]
        img_path = os.path.join(p, q, r)
        ori_img = cv2.imread(img_path)
        ori_img = cv2.resize(ori_img, (new_size, new_size))
        ori_img = ori_img.reshape((-1, ori_img.shape[-3], ori_img.shape[-2], ori_img.shape[-1]))

        p, q, r = mask_data[idx]
        img_path = os.path.join(p, q, r)
        mask_img = cv2.imread(img_path)

        mask_img = cv2.resize(mask_img, (new_size, new_size))
        ret, mask_img = cv2.threshold(mask_img, int(255 * 0.5), 255, cv2.THRESH_BINARY)
        mask_img_reshape = mask_img.reshape((-1, mask_img.shape[-3], mask_img.shape[-2], mask_img.shape[-1]))
        # img = direct_val(img)
        # print(img.shape)
        # print(mask_img.shape)
        img = image_to_tensor(img[0]).unsqueeze(0).cuda()
        mask_img_reshape = image_to_tensor(mask_img_reshape[0]).unsqueeze(0).cuda()
        ori_img = image_to_tensor(ori_img[0]).unsqueeze(0).cuda()

        img = img * mask_img_reshape + ori_img * (1 - mask_img_reshape)

        im = resize(img)
        # im = tensor_to_image(img_tensor_real[0])

        print(im.shape)

        # mask_img = tensor_to_image(mask_img[0])

        # im = Image.open(fold_path+img_path)
        # im = np.array(im)
        # original_image=im.copy()




        # mask_img = torch.Tensor(mask_img)
        # mask_img = mask_img.unsqueeze(0)
        # mask_img = mask_img.transpose(2, 3).transpose(1, 2)
        # mask_img = mask_img.to(device)


        with torch.no_grad():
            final_output = model(im)

        # plt.subplot(1,3,1)
        # plt.imshow(original_image)
        # plt.title('Original image')
        #
        # plt.subplot(1,3,2)
            print("Saved: {}".format('./result/' + r))
            final_image = tensor_to_image(final_output[0]) #(final_output[0][0] * 255).round()
            print(final_output.shape)
            img = tensor_to_image(im[0])
            img = cv2.resize(img, (new_size, new_size))
            cv2.imwrite('./tamper/' + r, img.astype(np.uint8))
            # torchvision.utils.save_image(im[0],
            #                              './tamper/' + r,
            #                              nrow=1, padding=0, normalize=False)
            cv2.imwrite('./result/copy-move/resize/' + r, final_image.astype(np.uint8))
            F1, TP = F1score(final_image[:,:,0], mask_img[:,:,0], thresh=0.5)
            if F1 > 0.4:
                F1_sum += F1
                valid += 1
            print("F1_sum {:3f} F1 {:3f}".format(F1_sum / (valid + 1e-3), F1))

    # plt.imshow((final_output[0][0]).cpu().detach(), cmap='gray')
    # plt.title('Predicted forgery mask')
    #
    # plt.subplot(1,3,3)
    # plt.imshow((final_output[0][0].cpu().detach().unsqueeze(2)>0.2)*torch.tensor(original_image))
    # plt.title('Suspicious regions detected')

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    #TN, TP, FN, FP
    result = [0, 0, 0 ,0]
    for row in range(height):
        for column in range(width):
            pixel = img[row, column]
            gt_pixel = gt_img[row, column]
            if pixel == gt_pixel:
                result[(pixel // 255)] += 1
            else:
                index = 2 if pixel == 0 else 3
                result[index] += 1
    return result



def getACC(TN, TP, FN, FP):
    return (TP+TN)/(TP+FP+FN+TN)
def getFPR(TN, FP):
    return FP / (FP + TN)
def getTPR(TP, FN):
    return TP/ (TP+ FN)
def getTNR(FP, TN):
    return TN/ (FP+ TN)
def getFNR(FN, TP):
    return FN / (TP + FN)
def getF1(TP, FP, FN):
    return (2*TP)/(2*TP+FP+FN)
def getBER(TN, TP, FN, FP):
    return 1/2*(getFPR(TN, FP)+FN/(FN+TP))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                # img_path = os.path.join(dirpath, fname)
                images.append((path, dirpath[len(path) + 1:], fname))
    assert images, '{:s} has no valid image file'.format(path)
    return images

def tensor_to_image(tensor):

    tensor = tensor * 255.0
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
    return np.clip(image, 0, 255).astype(np.uint8)

def F1score(predicted_binary, gt_image, thresh=0.2):
    # predicted_binary = tensor_to_image(predict_image[0])
    ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
    print(predicted_binary.shape)
    gt_image = gt_image
    print(gt_image.shape)
     #self.tensor_to_image(gt_image[0, :1, :, :])
    # ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)



    [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
    # print("{} {} {} {}".format(TN,TP,FN,FP))
    F1 = getF1(TP, FP, FN)
    # cv2.imwrite(save_path, predicted_binary)
    return F1, TP

class ForgeryDetector(nn.Module):
    
    # Model Initialization/Creation    
    def __init__(self,train_loader,detector=MantraNet(),lr=0.001):
        super(ForgeryDetector, self).__init__()
        
        self.detector=detector
        self.train_loader=train_loader
        self.cpt=-1
        self.lr=lr
        
    # Forward Pass of Model
    def forward(self, x):
        return self.detector(x)
    
    # Loss Function
    def loss(self, y_hat, y):
        return nn.BCELoss()(y_hat,y)

    # Optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.detector.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # return the list of optimizers and second empty list is for schedulers (if any)
        return [optimizer], []

    # Calls after prepare_data for DataLoader
    def train_dataloader(self):

        return self.train_loader

    
    # Training Loop
    def training_step(self, batch, batch_idx):
        # batch returns x and y tensors
        real_images, mask = batch
        B,_,_,_=real_images.size()
        self.cpt+=1

        predicted=self.detector(real_images).view(B,-1)
        mask=mask.view(B,-1)
  
        loss=self.loss(predicted,mask)

        self.log('BCELoss',loss,on_step=True,on_epoch=True,prog_bar=True)


        output = OrderedDict({
                'loss': loss,
                 })
           
        return output