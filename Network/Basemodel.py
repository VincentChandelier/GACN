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

        # dots = torch.triu(dots, diagonal=1) + torch.tril(dots, diagonal=-1)

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

        self.feature_map_to_tokens = Rearrange('b c h w -> b (h w)  c')

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Attention(N=N, heads=heads, dim_head=dim_head, dropout=dropout),
            )

        if self.downsample:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H // 2, w=W // 2)
        else:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H*2, w=W*2)

    def forward(self, x):
        x_conv = self.conv(x)
        downsample_x = self.feature_map_to_tokens_conv(x_conv)
        indentity = downsample_x

        x_tockens = self.feature_map_to_tokens(x)

        for attn in self.layers:
            downsample_x = attn(downsample_x, x_tockens) + downsample_x

        if self.depth:
            return self.reverse(downsample_x+indentity)
        else:
            return self.reverse(indentity)


class Basemodel(CompressionModel):
    def __init__(self, N=128, M=192, image_size=(384, 384), depth=[0, 0, 0, 0], heads=4, dim_head=192, dropout=0.1, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
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
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlockWithStride(N, M, stride=2),
            Convtransformer(H=H // 4, W=W // 4, N=M, fn=ResidualBlockWithStride(M, M, stride=2), depth=depth[0], heads=heads, dim_head=dim_head, dropout=0.,
                            downsample=True, ),
            Convtransformer(H=H // 8, W=W // 8, N=M, fn=ResidualBlockWithStride(M, M, stride=2), depth=depth[1], heads=heads, dim_head=dim_head, dropout=0.,
                            downsample=True, ),
        )

        self.g_s = nn.Sequential(
            Convtransformer(H=H // 16, W=W // 16, N=M, fn=ResidualBlockUpsample(M, M, 2), depth=depth[1], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),

            Convtransformer(H=H // 8, W=W // 8, N=M, fn=ResidualBlockUpsample(M, M, 2), depth=depth[0], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),
            ResidualBlockUpsample(M, N, 2),
            ResidualBlockUpsample(N, 3, 2),
        )

        self.h_a = nn.Sequential(
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            Convtransformer(H=H // 16, W=W // 16, N=M, fn=ResidualBlockWithStride(M, M, stride=2), depth=depth[2], heads=heads, dim_head=dim_head,
                            dropout=0., downsample=True, ),
            nn.LeakyReLU(inplace=True),
            Convtransformer(H=H // 32, W=W // 32, N=M, fn=ResidualBlockWithStride(M, M, stride=2), depth=depth[3], heads=heads, dim_head=dim_head,
                            dropout=0., downsample=True, ),
        )

        self.h_s = nn.Sequential(
            Convtransformer(H=H // 64, W=W // 64, N=M, fn=ResidualBlockUpsample(M, M, 2), depth=depth[3], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),
            nn.LeakyReLU(inplace=True),
            Convtransformer(H=H // 32, W=W // 32, N=M, fn=ResidualBlockUpsample(M, M, 2), depth=depth[2], heads=heads, dim_head=dim_head,
                            dropout=dropout, downsample=False),
            nn.LeakyReLU(inplace=True),
            conv(M, M * 3 // 2, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
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

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv



if __name__ == "__main__":
    # model = convTransformer(H=384, W=384, lenslet_num=8, viewsize=6, C=128, depth=4, heads=4, dim_head=96, mlp_dim=96, dropout=0.1, emb_dropout=0.)
    # model = convTransformer(H=192, W=192, channels=3, patchsize=2, dim=64, depth=2, heads=4,
    #                         dim_head=64, mlp_dim=64, dropout=0.1,
    #                         emb_dropout=0.)

    model = TestModel(N=128, M=128,  image_size=(384, 384), depth=[0, 0, 0, 0], heads=4, dim_head=192, dropout=0.1)
    # model = JointAutoregressiveHierarchicalPriors(192, 192)
    # model = Cheng2020Attention(128)
    input = torch.Tensor(1, 3, 384, 384)
    # from torchvision import models
    # model = models.resnet18()
    # print(model)
    out = model(input)
    flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)