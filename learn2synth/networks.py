from . import modules, utils
from torch import nn
from typing import Sequence


class SegNet(nn.Sequential):
    """A generic segmentation network that works with any backbone."""

    def __init__(self, ndim, in_channels, out_channels,
                 kernel_size=3, activation='Softmax',
                 backbone='UNet', kwargs_backbone=None):
        if isinstance(backbone, str):
            backbone_kls = globals()[backbone]
            backbone = backbone_kls(ndim, **(kwargs_backbone or {}))
        if activation and activation.lower() == 'softmax':
            activation = nn.Softmax(1)
        feat = modules.ConvBlock(ndim,
                                 in_channels, backbone.in_channels,
                                 kernel_size=kernel_size,
                                 activation=None)
        pred = modules.ConvBlock(ndim,
                                 backbone.out_channels, out_channels,
                                 kernel_size=1,
                                 activation=activation)
        super().__init__(feat, backbone, pred)


class UNet(nn.Module):
    """A highly parameterized U-Net (encoder-decoder + skip connections).

    conv ------------------------------------------(+)-> conv
         -down-> conv ---------------(+)-> conv -> up
                     -down-> conv -> up

    Reference
    ---------
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI (2015)
    https://arxiv.org/abs/1505.04597
    """

    def __init__(
            self,
            ndim: int,
            nb_features: Sequence[int] = (16, 24, 32, 48, 64, 96, 128, 192, 256, 320),
            nb_levels: int = 6,
            nb_conv: int = 2,
            kernel_size: int = 3,
            activation: str = 'ReLU',
            norm: str = 'instance',
            dropout: float = 0,
            residual: bool = False,
            factor: int = 2,
            use_strides: bool = False,
            order: str = 'cand',
            combine: str = 'cat',
    ):
        super().__init__()
        self.ndim = ndim
        self.nb_features = list(utils.ensure_list(nb_features, nb_levels))
        self.nb_levels = nb_levels
        self.nb_conv = nb_conv
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.residual = residual
        self.factor = factor
        self.use_strides = use_strides
        self.order = order
        self.combine = combine
        self.in_channels = self.out_channels = self.nb_features[0]

        # ── Encoder ───────────────────────────────────────────────────────────
        i, o = self.nb_features[0], self.nb_features[0]
        self.encoder = [self._conv_block(i, o)]
        for n in range(1, len(self.nb_features) - 1):
            i, o = self.nb_features[n - 1], self.nb_features[n]
            self.encoder += [modules.EncoderBlock(self._down_block(i, o),
                                                  self._conv_block(o))]
        if self.nb_levels > 1:
            i, o = self.nb_features[-2], self.nb_features[-1]
            self.encoder += [self._down_block(i, o)]
        self.encoder = nn.Sequential(*self.encoder)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = []
        for n in range(len(self.nb_features) - 1):
            i, o = self.nb_features[-n - 1], self.nb_features[-n - 2]
            m = i
            if self.combine == 'cat' and n > 0:
                i *= 2
            self.decoder += [modules.DecoderBlock(self._conv_block(i, m),
                                                  self._up_block(m, o))]
        i, o = self.nb_features[0], self.nb_features[0]
        if self.nb_levels > 1 and self.combine == 'cat':
            i *= 2
        self.decoder += [self._conv_block(i, o)]
        self.decoder = nn.Sequential(*self.decoder)

    # ── Builder helpers ───────────────────────────────────────────────────────

    def _conv_block(self, i, o=None):
        return modules.ConvGroup(self.ndim, i, o,
                                 activation=self.activation,
                                 kernel_size=self.kernel_size,
                                 order=self.order,
                                 nb_conv=self.nb_conv,
                                 norm=self.norm,
                                 residual=self.residual,
                                 dropout=self.dropout)

    def _down_block(self, i, o=None):
        if self.use_strides:
            return modules.StridedConvBlockDown(self.ndim, i, o,
                                                activation=self.activation,
                                                strides=self.factor,
                                                order=self.order,
                                                kernel_size=self.factor,
                                                norm=self.norm,
                                                dropout=self.dropout)
        return modules.ConvBlockDown(self.ndim, i, o,
                                     activation=self.activation,
                                     factor=self.factor,
                                     order=self.order,
                                     kernel_size=1,
                                     norm=self.norm,
                                     dropout=self.dropout)

    def _up_block(self, i, o=None):
        if self.use_strides:
            return modules.StridedConvBlockUp(self.ndim, i, o,
                                              activation=self.activation,
                                              strides=self.factor,
                                              order=self.order,
                                              kernel_size=self.factor,
                                              norm=self.norm,
                                              dropout=self.dropout,
                                              combine=self.combine)
        return modules.ConvBlockUp(self.ndim, i, o,
                                   activation=self.activation,
                                   factor=self.factor,
                                   order=self.order,
                                   kernel_size=1,
                                   norm=self.norm,
                                   dropout=self.dropout,
                                   combine=self.combine)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        nb_levels = len(self.encoder)
        if any(s < 2 ** nb_levels for s in x.shape[2:]):
            raise ValueError(
                f'UNet with {nb_levels} levels requires spatial dimensions '
                f'>= {2 ** nb_levels}, got {list(x.shape[2:])}'
            )

        # Encoder — build skip connections
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Decoder — upsample + merge skips
        x = skips.pop(-1)
        for n in range(len(self.decoder) - 1):
            x = self.decoder[n].conv(x)
            x = self.decoder[n].up(x, skips.pop(-1))

        return self.decoder[-1](x)
