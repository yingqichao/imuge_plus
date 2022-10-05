import torch
import torch.nn as nn
from models.invertible_net import HaarUpsampling, HaarDownsampling
# from lama_models.ffc import SpectralTransform, FF


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class HalfFourierBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, spectral_pos_encoding=False, fft_norm='ortho'):
        # bn_layer not used
        super(HalfFourierBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv_local = depthwise_separable_conv(nin=in_channels//2,nout=out_channels, kernels_per_layer=2,
                                                   kernel_size=3, stride=1, padding=1)
        self.ln_local = LayerNorm2d(out_channels//2)

        self.conv_layer = depthwise_separable_conv(nin=in_channels + (2 if spectral_pos_encoding else 0),
                                          nout=out_channels*2, kernels_per_layer=2,
                                          kernel_size=3, stride=1, padding=1)
        self.ln = LayerNorm2d(out_channels)
        self.simple_gate = SimpleGate()

        self.spectral_pos_encoding = spectral_pos_encoding

        self.fft_norm = fft_norm

        self.sca_layer = SimplifiedChannelAttention(out_channels)
        self.conv_final = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, input):
        x, identity_path = torch.chunk(input, 2, dim=1)
        #### fourier path ####
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.simple_gate(ffted)
        ffted = self.ln(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        ### local path ###
        identity_conv = self.conv_local(identity_path)
        identity_conv = self.simple_gate(identity_conv)
        identity_conv = self.ln_local(identity_conv)

        ### 1x1 conv to fuse features ###
        out = torch.cat([output, identity_conv], dim=1)
        hwa_out = self.sca_layer(out)
        hwa_out = self.conv_final(hwa_out)
        return input + hwa_out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, nin):
        super(SimplifiedChannelAttention, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        attn = self.sca(x)
        return x*attn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer, kernel_size=3, padding=1, stride=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin, stride=1)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False, subtask=0):
        super(SKFF, self).__init__()
        self.subtask = subtask
        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[0].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

class elastic_layer(nn.Module):
    def __init__(self, nin, nch=16, depth=4, nout=None, activate_last=False, bili_downs=None, bili_ups=None,
                 feature_split_ratio=0.75):
        super(elastic_layer, self).__init__()
        self.nch = nch
        self.depth = depth
        self.feature_split_ratio = feature_split_ratio
        self.activate_last = activate_last
        self.ch_split = int(self.nch * (1 - self.feature_split_ratio))
        if bili_downs is None:
            self.bili_downs = nn.ModuleList([])
            self.bili_downs.append(nn.Identity())
            ### 1,0.5,0.25,0.125
            for i in range(1, self.depth):
                self.bili_downs.append(bili_resize(0.5 ** (i)))
        else:
            self.bili_downs = bili_downs

        if bili_ups is None:
            self.bili_ups = nn.ModuleList([])
            self.bili_ups.append(nn.Identity())
            ### 1,2,4,8
            for i in range(1, self.depth):
                self.bili_ups.append(bili_resize(2 ** (i)))
        else:
            self.bili_ups = bili_ups

        self.hwa_blocks = nn.ModuleList([])
        self.depth = depth
        for i in range(self.depth):
            self.hwa_blocks.append(HalfFourierBlock(in_channels=nch))

        if not self.activate_last:
            self.skff_blocks = nn.ModuleList([])
            for i in range(0, self.depth):
                if i==0:
                    self.skff_blocks.append(nn.Identity())
                else:
                    self.skff_blocks.append(SKFF(in_channels=self.nch-self.ch_split, height=i + 1))
        else:
            self.skff_blocks = nn.Sequential(*[
                SKFF(in_channels=nch, height=self.depth),
                depthwise_separable_conv(nin=nch,nout=nout,kernels_per_layer=1),
            ])


    def forward(self, x_feats: list):
        if not self.activate_last:
            ## hwa block
            x_hwas_shares, x_hwas_identitys = [], []
            for i_scale in range(self.depth):
                x_hwas_feat = self.hwa_blocks[i_scale](x_feats[i_scale])

                x_hwas_identity, x_hwas_merge = x_hwas_feat[:,:self.ch_split], x_hwas_feat[:,self.ch_split:]
                x_hwas_identitys.append(x_hwas_identity)
                x_hwas_shares.append(x_hwas_merge)
            ## skff block

            x_skff = []
            for i_scale in range(self.depth):
                if i_scale==0:
                    x_skff.append(torch.cat([x_hwas_identitys[i_scale], x_hwas_shares[i_scale]],dim=1))
                else:
                    x_hwas_identity = x_hwas_identitys[i_scale]
                    skff_input = []
                    for j_scale in range(0, i_scale + 1):
                        input_x_hwas = self.bili_ups[i_scale - j_scale](x_hwas_shares[j_scale])
                        skff_input.append(input_x_hwas)
                    skff_out = self.skff_blocks[i_scale](skff_input)
                    x_skff.append(torch.cat([x_hwas_identity, skff_out],dim=1))

            return x_skff

        else:
            ## hwa block
            x_hwas = []
            for i_scale in range(self.depth):
                x_hwas_feat = self.hwa_blocks[i_scale](x_feats[i_scale])
                x_hwas.append(x_hwas_feat)

            skff_input = []
            for j_scale in range(0, self.depth):
                input_x_hwas = self.bili_ups[self.depth-1 - j_scale](x_hwas[j_scale])
                skff_input.append(input_x_hwas)
            x_skff = self.skff_blocks(skff_input)

        return x_skff

class my_own_elastic(nn.Module):
    def __init__(self, nin, nch=16, depth=4, nout=None, num_blocks=8):
        super(my_own_elastic, self).__init__()
        if nout is None:
            nout = nin
        self.nin = nin
        self.nout = nout
        self.depth = depth
        self.num_blocks = num_blocks
        self.init_convs = nn.ModuleList([])
        self.elastic_layers = nn.ModuleList([])
        self.bili_downs = nn.ModuleList([])
        self.bili_downs.append(nn.Identity())
        ### 1,0.5,0.25,0.125
        for i in range(1, self.depth):
            self.bili_downs.append(bili_resize(0.5 ** (i)))

        self.bili_ups = nn.ModuleList([])
        self.bili_ups.append(nn.Identity())
        ### 1,2,4,8
        for i in range(1, self.depth):
            self.bili_ups.append(bili_resize(2 ** (i)))

        for i in range(self.num_blocks):
            self.init_convs.append(
                nn.Sequential(*[
                    depthwise_separable_conv(nin=nin,nout=nch*2,kernels_per_layer=16),
                    SimpleGate(),
                              ]
                )
            )

            self.elastic_layers.append(elastic_layer(nin, nch, depth, nout,
                                                     bili_downs=self.bili_downs, bili_ups=self.bili_ups,
                                                     activate_last=(i==self.num_blocks-1)))



    def forward(self, X):
        ### decomposit X into hiearchical input ###
        batch, channel, H, W = X.shape
        x_inits = []
        x_previous = self.bili_downs[self.depth-1](X)
        x_inits.append(x_previous)
        for i in range(1,self.depth):
            x_current_gt = self.bili_downs[self.depth-1-i](X)
            x_previous = self.bili_ups[1](x_previous)
            x_diff = x_current_gt-x_previous
            x_inits.append(x_diff)
            x_previous = x_current_gt

        ### init
        x_feats = []
        for i_scale in range(self.depth):
            x_feats.append(self.init_convs[i_scale](x_inits[i_scale]))

        ## elastic layers
        for i_layers in range(self.num_blocks):
            x_feats = self.elastic_layers[i_layers](x_feats)

        return x_feats if self.nin!=self.nout else X+x_feats



if __name__ == '__main__':
    # model = HalfFourierBlock(in_channels=16, out_channels=16)
    X = torch.randn(1, 3, 256,256)

    # model = SKFF(in_channels=16)
    # X = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64)]
    print(X.shape)

    model = my_own_elastic(nin=3,nout=1, depth=4, num_blocks=8)
    Y = model(X)
    print(Y.shape)
    from thop import profile
    flops, params = profile(model, (X,))
    print(flops)
    print(params)