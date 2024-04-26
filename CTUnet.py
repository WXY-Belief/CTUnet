import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch import einsum
from einops import rearrange

flag = False

"""---------------CNN-------------------"""
class CnnBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CnnBlock, self).__init__()
        self.Conv_BN_Relu_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.Conv_BN_Relu_2(x)
        return out


"""---------------MIT Transformer-------------------"""
class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(self,dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)

        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MIT(nn.Module):
    def __init__(self, in_ch, out_ch, head: int = 1, ff_expansion: int = 8, reduction_ratio: int = 8,
                 num_layer: int = 2, kernel_para: tuple = (3, 2, 1)):
        super().__init__()
        self.kernel = kernel_para[0]
        self.stride = kernel_para[1]
        self.padding = kernel_para[2]
        self.get_overlap_patches = nn.Unfold(kernel_size=kernel_para[0], stride=kernel_para[1], padding=kernel_para[2])
        self.overlap_patch_embed = nn.Conv2d(in_ch * kernel_para[0] ** 2, out_ch, 1)

        self.layers = nn.ModuleList([])

        for _ in range(num_layer):
            self.layers.append(nn.ModuleList([
                PreNorm(out_ch, EfficientSelfAttention(dim=out_ch, heads=head, reduction_ratio=reduction_ratio)),
                PreNorm(out_ch, MixFeedForward(dim=out_ch, expansion_factor=ff_expansion)),
            ]))

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
                nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)

        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_patch_embed(x)
        for (attn, ff) in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.upsample(x)
        return x

"""---------------CBAM-------------------"""
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, out_channal: int=1):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, out_channal, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

"""---------------CTBblock-------------------"""
class one_CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(one_CBAM, self).__init__()
        self.channel_merge = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(gate_channels, gate_channels, kernel_size=1),
        )
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.channel_merge(x) + x
        x_out = self.ChannelGate(x_out)
        x_out = self.SpatialGate(x_out)
        return x_out

class CTBlock(nn.Module):
    def __init__(self, cnn, mit, in_channel, out_channel):
        super(CTBlock, self).__init__()
        self.mit = mit
        self.cnn = cnn
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.on_CBAM = one_CBAM(out_channel)

    def forward(self, x):
        mit_out = self.mit(x)
        cnn_out = self.cnn(x)
        merge_result = self.on_CBAM(self.merge(torch.cat([mit_out, cnn_out], dim=1)))
        down_result = self.downsample(merge_result)

        return (merge_result, mit_out, cnn_out), down_result

"""---------------CTFusion-------------------"""
class CTFusion(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CTFusion, self).__init__()

        self.Residual = nn.Sequential(
            nn.Conv2d(gate_channels * 3, gate_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(gate_channels*2),
            nn.ReLU(),
            nn.Conv2d(gate_channels*2, gate_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(),
        )

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
    def forward(self, X):
        c_out = self.ChannelGate(X[2]) + X[2]
        s_out = self.SpatialGate(X[1]) + X[1]
        out = self.Residual(torch.cat((c_out, s_out, X[0]),dim=1))

        return out

class Upsamplelayer(nn.Module):
    def __init__(self, in_ch, out_ch, flag: bool=False):
        super(Upsamplelayer, self).__init__()
        self.Conv_BN_Relu_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_ch/2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(in_ch/2), out_channels=int(in_ch/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_ch/2)),
            nn.ReLU()
        )
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=int(in_ch/2), out_channels=out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.flag = flag
    def forward(self, UP_x, F_X):
        x = torch.cat((F_X, UP_x), dim=1)
        if self.flag:
            return self.Conv_BN_Relu_2(x)
        else:
            out = self.Upsample(self.Conv_BN_Relu_2(x))

        return out

class CTUnet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: tuple = (64, 128, 256, 512), num_class: int = 3,
                 heads: tuple = (1, 2, 4, 8), ff_expansions: tuple = (8, 8, 4, 4),
                 reduction_ratios: tuple = (8, 4, 2, 1), num_layer: int = 1, kernel_para=(3, 2, 1)):
        super(CTUnet, self).__init__()

        self.CT_block_1 = CTBlock(CnnBlock(in_ch, out_ch[0]),
                                        MIT(in_ch=in_ch, out_ch=out_ch[0], head=heads[0],
                                            ff_expansion=ff_expansions[0],
                                            reduction_ratio=reduction_ratios[0],
                                            num_layer=num_layer, kernel_para=kernel_para), out_ch[0] * 2, out_ch[0])
        self.CT_block_2 = CTBlock(CnnBlock(out_ch[0], out_ch[1]),
                                        MIT(in_ch=out_ch[0], out_ch=out_ch[1], head=heads[1],
                                            ff_expansion=ff_expansions[1], reduction_ratio=reduction_ratios[1],
                                            num_layer=num_layer, kernel_para=kernel_para), out_ch[1] * 2, out_ch[1])
        self.CT_block_3 = CTBlock(CnnBlock(out_ch[1], out_ch[2]),
                                        MIT(in_ch=out_ch[1], out_ch=out_ch[2], head=heads[2],
                                            ff_expansion=ff_expansions[2], reduction_ratio=reduction_ratios[2],
                                            num_layer=num_layer, kernel_para=kernel_para), out_ch[2] * 2, out_ch[2])
        self.CT_block_4 = CTBlock(CnnBlock(out_ch[2], out_ch[3]),
                                        MIT(in_ch=out_ch[2], out_ch=out_ch[3], head=heads[3],
                                            ff_expansion=ff_expansions[3],
                                            reduction_ratio=reduction_ratios[3],
                                            num_layer=num_layer, kernel_para=kernel_para), out_ch[3] * 2, out_ch[3])
        self.peak = CTBlock(CnnBlock(out_ch[3], out_ch[3]),
                                        MIT(in_ch=out_ch[3], out_ch=out_ch[3], head=heads[3],
                                            ff_expansion=ff_expansions[3],
                                            reduction_ratio=reduction_ratios[3],
                                            num_layer=num_layer, kernel_para=kernel_para), out_ch[3] * 2, out_ch[3])

        self.peak_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch[3], out_channels=out_ch[3], kernel_size=4, stride=2,padding=1))

        self.CTFusion_1 = CTFusion(out_ch[0])
        self.CTFusion_2 = CTFusion(out_ch[1])
        self.CTFusion_3 = CTFusion(out_ch[2])
        self.CTFusion_4 = CTFusion(out_ch[3])

        self.UpSample_1 = Upsamplelayer(out_ch[1], out_ch[0], flag=True)
        self.UpSample_2 = Upsamplelayer(out_ch[1] * 2, out_ch[0])
        self.UpSample_3 = Upsamplelayer(out_ch[2] * 2, out_ch[1])
        self.UpSample_4 = Upsamplelayer(out_ch[3] * 2, out_ch[2])

        self.decode_head = nn.Sequential(
            nn.Conv2d(in_channels=out_ch[0], out_channels=num_class, kernel_size=1),
        )

    def forward(self, x):
        layer_1, d1 = self.CT_block_1(x)
        if flag:
            print("layer_1:", d1.shape)
        layer_2,d2 = self.CT_block_2(d1)
        if flag:
            print("layer_2:", d2.shape)
        layer_3,d3 = self.CT_block_3(d2)
        if flag:
            print("layer_3:", d3.shape)
        layer_4,d4 = self.CT_block_4(d3)
        if flag:
            print("layer_4:", d4.shape)
        (peak,_,_), _ = self.peak(d4)
        peak = self.peak_up(peak)
        if flag:
            print("peak", peak.shape)

        F4 = self.CTFusion_4(layer_4)
        if flag:
            print("F4:", F4.shape)
        F3 = self.CTFusion_3(layer_3)
        if flag:
            print("F3:", F3.shape)
        F2 = self.CTFusion_2(layer_2)
        if flag:
            print("F2:", F2.shape)
        F1 = self.CTFusion_1(layer_1)
        if flag:
            print("F1:", F1.shape)

        UP4 = self.UpSample_4(peak, F4)
        if flag:
            print("UP4:", UP4.shape)
        UP3 = self.UpSample_3(UP4, F3)
        if flag:
            print("UP3:", UP3.shape)
        UP2 = self.UpSample_2(UP3, F2)
        if flag:
            print("UP2:", UP2.shape)
        UP1 = self.UpSample_1(UP2, F1)
        if flag:
            print("UP1:", UP1.shape)

        out = self.decode_head(UP1)
        return out


if __name__ == "__main__":
    image = torch.randn((2, 3, 512, 512))
    print(image.shape)

    model = CTUnet()
    out = model(image)
    print(out.shape)
