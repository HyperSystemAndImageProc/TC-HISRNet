import torch
import torch.nn as nn
import math

from torch import nn
import torch
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out

class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=64, dim=512, patch_size=8):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        self.conv2 = nn.Conv2d(in_channel*2, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)


    def forward(self, x):
        y = x.clone()  # bs,c,h,w

        ## Local Representation
        y = self.conv2(x)  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w

        return y


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class RAF(nn.Module):
    def __init__(self, n_feats, kernel_size=3, act=nn.ReLU, res_scale=1, conv=default_conv):
        super(RAF, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(TwoCNN, self).__init__()

        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1)))

    def forward(self, x):
        out = self.body(x)
        out = torch.add(out, x)

        return out


class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)

    def forward(self, x):
        out = x
        for i in range(2):

            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)

        out = torch.add(out, x)
        return out





class TCHISRNet(nn.Module):
    def __init__(self, args):
        super(TCHISRNet, self).__init__()

        scale = 2
        n_feats = args.n_feats

        self.n_module = 5

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.gamma_X = nn.Parameter(torch.ones(self.n_module))
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module))
        self.gamma_DFF = nn.Parameter(torch.ones(2))
        self.gamma_FCF = nn.Parameter(torch.ones(2))

        ThreeHead = []
        ThreeHead.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        ThreeHead.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = []
        TwoHead.append(wn(nn.Conv2d(1, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        self.TwoHead = nn.Sequential(*TwoHead)

        TwoTail = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats * 4, kernel_size=(3, 3), stride=1, padding=(1, 1))))
                TwoTail.append(nn.PixelShuffle(2))
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats * 9, kernel_size=(3, 3), stride=1, padding=(1, 1))))
            TwoTail.append(nn.PixelShuffle(3))

        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        self.TwoTail = nn.Sequential(*TwoTail)

        twoCNN = []
        for _ in range(self.n_module):
            twoCNN.append(TwoCNN(wn, n_feats))
        self.twoCNN = nn.Sequential(*twoCNN)

        self.reduceD_Y = wn(nn.Conv2d(n_feats * self.n_module, n_feats, kernel_size=(1, 1), stride=1))
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1)))

        threeCNN = []
        for _ in range(self.n_module):
            threeCNN.append(ThreeCNN(wn, n_feats))
        self.threeCNN = nn.Sequential(*threeCNN)

        reduceD = []
        for _ in range(self.n_module):
            reduceD.append(wn(nn.Conv2d(n_feats * 4, n_feats, kernel_size=(1, 1), stride=1)))
        self.reduceD = nn.Sequential(*reduceD)

        self.reduceD_X = wn(nn.Conv3d(n_feats * self.n_module, n_feats, kernel_size=(1, 1, 1), stride=1))

        self.SA = RAF(n_feats=n_feats*3, kernel_size=3, act=nn.ReLU(True), res_scale=1)

        threefusion = []
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats * 4, n_feats, kernel_size=(1, 1), stride=1))
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1, 1), stride=1))

        self.reduceD_FCF = wn(nn.Conv2d(n_feats * 2, n_feats, kernel_size=(1, 1), stride=1))
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1, 1), stride=1))

        self.m = MobileViTAttention()

    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)
        x = self.ThreeHead(x)


        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []

        for j in range(self.n_module):
            x = self.threeCNN[j](x)

            channelX.append(self.gamma_X[j] * x)

            z1 = x.reshape(x.shape[0], -1, x.shape[-1], x.shape[-1])

            z1 = self.SA(z1)

            y = self.twoCNN[j](y)
            y = torch.cat([y, z1], 1)
            y = self.reduceD[j](y)

            channelY.append(self.gamma_Y[j] * y)

        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)

        y = torch.cat(channelY, 1)
        y = self.reduceD_Y(y)
        y = self.twofusion(y)


        z1 = x.reshape(x.shape[0], -1, x.shape[-1], x.shape[-1])

        y = torch.cat([self.gamma_DFF[0] * z1, self.gamma_DFF[1] * y], 1)
        y = self.reduceD_DFF(y)


        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0] * y, self.gamma_FCF[1] * localFeats], 1)
            y = self.m(y)
            localFeats = y
        y = torch.add(y, skip_y)
        y = self.TwoTail(y)
        y = y.squeeze(1)

        return y, localFeats

