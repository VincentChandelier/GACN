"""
A simple test algorithm to rewrite the network to test how to use the code
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    MaskedConv2d,
    GDN,
)
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

from ptflops import get_model_complexity_info
import math
from compressai.models.priors import CompressionModel, GaussianConditional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


# classes
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class postNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """

    N: channels of input feature maps
    heads: multi heads of layers
    dim_head: each dim of heads
    """

    def __init__(self, N=128, heads=2, dim_head=128, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == N)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(N, inner_dim, bias=False)
        self.q_reshape = Rearrange('b n (h d) -> b h n d', h=self.heads)

        self.to_kv = nn.Linear(N, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, N),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, downsample_x, x):
        """

        Args:
            downsample_x: # N H/2*W/2 C
            x: # N H*W C

        Returns:

        """

        q = self.to_q(downsample_x)
        q = self.q_reshape(q)

        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Convtransformer(nn.Module):
    """
    H: height of input feature maps
    W: width of input feature maps
    N: channels of input feature maps
    heads: multi heads of layers
    dim_head: each dim of heads
    """

    def __init__(self, H, W, N, fn, depth=2, heads=2, dim_head=128, dropout=0., downsample=True, ):
        super(Convtransformer, self).__init__()
        self.depth = depth
        self.downsample = downsample

        self.conv = fn
        self.feature_map_to_tokens_conv = Rearrange('b c h w -> b (h w)  c')
        self.down_norm = nn.LayerNorm(N)

        self.feature_map_to_tokens = Rearrange('b c h w -> b (h w)  c')
        self.norm = nn.LayerNorm(N)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(N=N, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(N),
                postNorm(N, FeedForward(N, N, dropout=dropout)),
            ]))

        if self.downsample:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H // 2, w=W // 2)
        else:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H*2, w=W*2)

    def forward(self, x):
        x_conv = self.conv(x)
        downsample_x = self.feature_map_to_tokens_conv(x_conv)
        downsample_x = self.down_norm(downsample_x)
        indentity = downsample_x

        x_tockens = self.feature_map_to_tokens(x)
        x_tockens_norm = self.norm(x_tockens)

        for attn, norm, ff in self.layers:
            downsample_x = attn(downsample_x, x_tockens_norm) + downsample_x
            downsample_x = norm(downsample_x) + downsample_x
            downsample_x = ff(downsample_x) + downsample_x

        if self.depth:
            return self.reverse(downsample_x+indentity)
        else:
            return self.reverse(indentity)


class HyperGlobalModule(CompressionModel):

    def __init__(self, N=128, M=192, image_size=(384, 384), depth=[2, 0, 2, 0], heads=4, dim_head=128, dropout=0.1, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        H, W = pair(image_size)
        """
             N: channel number of main network
             M: channnel number of latent space
             scale_factor: first stage of transform
             imagesize: input image size
             patchsize: split the image into patch for transform
             lenslet: the number of views in a patch
             dim:
             fdepth: first depth of the transformer
             sdepth: second depth of the transformer
             dim: the dim of attention 
             heads: the heads Multi-head attention
             dim_head: 
        """
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            Convtransformer(H=H // 4, W=W // 4, N=N, fn=conv(N, N), depth=depth[0], heads=heads, dim_head=dim_head, dropout=0.,
                            downsample=True, ),
            GDN(N),
            conv(N, M)
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            Convtransformer(H=H // 8, W=W // 8, N=N, fn=deconv(N, N), depth=depth[0], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            Convtransformer(H=H // 16, W=W // 16, N=N, fn=conv(N, N), depth=depth[2], heads=heads, dim_head=dim_head,
                            dropout=0., downsample=True, ),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            Convtransformer(H=H // 32, W=W // 32, N=N, fn=deconv(N, N), depth=depth[2], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



if __name__ == "__main__":
    # model = convTransformer(H=384, W=384, lenslet_num=8, viewsize=6, C=128, depth=4, heads=4, dim_head=96, mlp_dim=96, dropout=0.1, emb_dropout=0.)
    # model = convTransformer(H=192, W=192, channels=3, patchsize=2, dim=64, depth=2, heads=4,
    #                         dim_head=64, mlp_dim=64, dropout=0.1,
    #                         emb_dropout=0.)

    model = TestModel(N=128, M=192,  image_size=(384, 384), depth=[2, 0, 2, 0], heads=4, dim_head=128, dropout=0.1)
    # model = JointAutoregressiveHierarchicalPriors(192, 192)
    # model = Cheng2020Attention(128)
    input = torch.Tensor(1, 3, 384, 384)
    # from torchvision import models
    # model = models.resnet18()
    # print(model)
    out = model(input)
    flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
