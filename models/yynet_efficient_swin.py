import torch
import torch.nn as nn
from config import swin_tiny_patch4_224_2 as swin
import torch.nn.functional as F
import math
import DFConv
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation blocks  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class cSE(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        input_x = x
        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        return torch.mul(input_x, x)


class sSE(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        return torch.mul(input_x, x)


class scSE(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        return torch.add(self.cse_block(x), self.sse_block(x))


class cSE1(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor):
        input_x = x
        x_max = self.maxpool(x).view(*(x.shape[:-2]))
        x_max = F.relu(self.linear1(x_max), inplace=True)
        x_max = self.linear2(x_max)

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)

        x = torch.add(x, x_max)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        return torch.mul(input_x, x)


class sSE1(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int):
        super().__init__()
        self.DWconv1 = DWconv(in_channels, 1, stride=1, padding=4, dilation=4)
        self.DWconv2 = DWconv(in_channels, 1, stride=1, padding=6, dilation=6)
        self.conv1   = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2   = nn.Conv2d(3, 1, 1, 1)

    def forward(self, x: torch.Tensor):
        input_x = x
        x1 = self.conv1(x)
        x2 = self.DWconv1(x)
        x3 = self.DWconv2(x)
        x  = self.conv2(torch.cat([x1, x2, x3], dim=1))
        x  = torch.sigmoid(x)
        return torch.mul(input_x, x)


class scSE1(nn.Module):  # noqa: N801
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE1(in_channels)

    def forward(self, x: torch.Tensor):
        return torch.add(self.cse_block(x), self.sse_block(x))


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise-Separable Conv  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1):
        super(DWconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                    stride=stride, padding=padding,
                                    dilation=dilation, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                    stride=1, padding=0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# HeightEncoder  ← NEW
#
# Converts a single-channel greyscale height raster (B, 1, H, W) into 4
# multi-scale feature maps that spatially match the CNN stages:
#
#   h1  →  H/4   × W/4    matches  x_u64   fed into  up_c_3_1
#   h2  →  H/8   × W/8    matches  x_u_2   fed into  up_c_2_1
#   h3  →  H/16  × W/16   matches  x_u_3   fed into  up_c_1_1
#   h4  →  H/32  × W/32   matches  x_u     fed into  up_c
#
# Height values are continuous scalars so we keep the stem simple (no sigmoid).
# out_ch=16 is a lightweight choice; change _H_CH below to tune.
# ─────────────────────────────────────────────────────────────────────────────

_H_CH = 16   # number of channels produced by HeightEncoder at every scale


class HeightEncoder(nn.Module):
    def __init__(self, out_ch: int = _H_CH):
        super().__init__()
        # stem: 1 → out_ch,  stride 2  →  H/2
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # s1 → H/4,  s2 → H/8,  s3 → H/16,  s4 → H/32
        self.s1 = self._stage(out_ch, out_ch)
        self.s2 = self._stage(out_ch, out_ch)
        self.s3 = self._stage(out_ch, out_ch)
        self.s4 = self._stage(out_ch, out_ch)

    @staticmethod
    def _stage(in_ch, out_ch):
        """stride-2 conv to halve spatial size + extra 3×3 for local context."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, height_map: torch.Tensor):
        """
        Args:
            height_map: (B, 1, H, W) — greyscale height raster,
                        recommended to normalise to [0, 1] before calling.
        Returns:
            h1, h2, h3, h4  — (B, out_ch, H/4, W/4) … (B, out_ch, H/32, W/32)
        """
        x  = self.stem(height_map)
        h1 = self.s1(x)
        h2 = self.s2(h1)
        h3 = self.s3(h2)
        h4 = self.s4(h3)
        return h1, h2, h3, h4


# ─────────────────────────────────────────────────────────────────────────────
# ICEF — Inter-modal Cross-attention Enhancement Fusion
#
# ch_1 now includes the _H_CH height channels that are cat-ed onto the
# CNN feature before this module is called. Everything else is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class ICEF(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, drop_rate=0.):
        super(ICEF, self).__init__()

        self.ch_out    = ch_out
        self.drop_rate = drop_rate
        self.softmax   = nn.Softmax(dim=-1)
        self.sigmoid   = nn.Sigmoid()
        self.relu      = nn.ReLU(inplace=True)

        self.scse_cnn  = scSE1(ch_1)
        self.scse_attn = scSE1(ch_2)

        self.dropout   = nn.Dropout2d(drop_rate)

        self.q1 = DWconv(ch_1, ch_out // 2)
        self.q2 = DWconv(ch_2, ch_out // 2)
        self.k1 = DWconv(ch_1, ch_out)
        self.k2 = DWconv(ch_2, ch_out)

        self.dw1 = DWconv(ch_1, ch_1, padding=6, dilation=6)
        self.dw1_1 = nn.Sequential(
            DWconv(ch_1, ch_1),
            nn.BatchNorm2d(ch_1),
            nn.ReLU()
        )

        self.dw2 = DWconv(ch_2, ch_2, padding=6, dilation=6)
        self.dw2_2 = nn.Sequential(
            DWconv(ch_2, ch_2),
            nn.BatchNorm2d(ch_2),
            nn.ReLU()
        )

        self.residual = nn.Sequential(
            nn.BatchNorm2d(ch_1 + ch_2),
            nn.ReLU(),
            nn.Conv2d(ch_1 + ch_2, ch_out, 3, 1, 1)
        )

    def forward(self, g, x):
        m_batchsize1, C1, height1, width1 = g.size()
        m_batchsize2, C2, height2, width2 = x.size()

        c1   = self.scse_cnn(g)
        c1   = self.dw1(c1)
        c1   = self.sigmoid(c1) * c1
        c1_v = c1.view(m_batchsize1, C1, height1 * width1)

        q1 = self.q1(g)
        k1 = self.k1(g).view(m_batchsize1, self.ch_out, height1 * width1)

        A1   = self.scse_attn(x)
        A1   = self.dw2(A1)
        A1   = self.sigmoid(A1) * A1
        A1_v = A1.view(m_batchsize2, C2, height2 * width2)

        q2 = self.q2(x)
        k2 = self.k2(x).view(m_batchsize2, self.ch_out, height2 * width2)

        q       = torch.cat([q1, q2], dim=1).view(m_batchsize2, self.ch_out, height2 * width2).permute(0, 2, 1)
        energy1 = torch.bmm(q, k1)
        energy2 = torch.bmm(q, k2)
        att1    = self.softmax(energy1)
        att2    = self.softmax(energy2)

        c = torch.bmm(c1_v, att1.permute(0, 2, 1))
        c = c.view(m_batchsize1, C1, height1, width1)
        c = torch.add(c, g)
        c = self.dw1_1(c)
        c = torch.add(c, c1)

        A = torch.bmm(A1_v, att2.permute(0, 2, 1))
        A = A.view(m_batchsize2, C2, height2, width2)
        A = torch.add(A, x)
        A = self.dw2_2(A)
        A = torch.add(A, A1)

        fuse = self.residual(torch.cat([A, c], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        return fuse


# ─────────────────────────────────────────────────────────────────────────────
# ACO — Adaptive Context-aware Output decoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ACO(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, if_PFN=False):
        super().__init__()
        self.ch_1     = ch_1
        self.pfn      = if_PFN
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.channel_att = cSE1(ch_1 + ch_2)
        self.conv1 = nn.Sequential(
            DWconv(ch_1 + ch_2, ch_1 + ch_2),
            nn.BatchNorm2d(ch_1 + ch_2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            DWconv(ch_1 + ch_2, ch_1 + ch_2),
            nn.BatchNorm2d(ch_1 + ch_2),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(ch_1 + ch_2, ch_out, kernel_size=3, stride=1, padding=1)

        if self.pfn:
            self.final_mask = nn.Sequential(
                Conv(ch_1 + ch_2, ch_1 // 4, 3, bn=True, relu=True),
                Conv(ch_1 // 4, 1, 3, bn=False, relu=False)
            )
            self.final_boundary = nn.Sequential(
                Conv(ch_1 + ch_2, ch_1 // 4, 3, bn=True, relu=True),
                Conv(ch_1 // 4, 1, 3, bn=False, relu=False)
            )

        self.act = nn.ReLU()

    def forward(self, b, f):
        b_up  = self.upsample(b)
        x_cat = torch.cat([b_up, f], dim=1)
        x     = self.channel_att(x_cat)

        x_b = self.conv1(x)
        x_b = torch.add(x_cat, x_b)

        x_f = self.conv2(x)
        x_f = torch.add(x_cat, x_f)

        if self.pfn:
            mask     = self.final_mask(x_b)
            boundary = self.final_boundary(x_b)
            x_f      = self.conv3(x_f)
            return x_f, mask, boundary

        return self.conv3(x_f)


# ─────────────────────────────────────────────────────────────────────────────
# CTCFNet  ← updated with HeightEncoder + height fusion at all 4 ICEF scales
# ─────────────────────────────────────────────────────────────────────────────

class CTCFNet(nn.Module):
    def __init__(self, num_classes=6, drop_rate=0.4, normal_init=True, pretrained=False):
        super(CTCFNet, self).__init__()

        # ── EfficientNetV2-RW-T backbone ────────────────────────────────────
        self.efficienet = timm.create_model('efficientnetv2_rw_t', num_classes=0)
        self.act1 = nn.SiLU()
        if pretrained:
            self.efficienet = timm.create_model('efficientnetv2_rw_t.ra2_in1k', num_classes=0)

        # ── Swin Transformer backbone ───────────────────────────────────────
        self.transformer = swin(pretrained=pretrained)

        # ── Building Height Encoder  ← NEW ─────────────────────────────────
        # Produces (B, _H_CH, *) at 4 scales matching each CNN feature stage.
        self.height_encoder = HeightEncoder(out_ch=_H_CH)

        self.extract_features = {}

        # ── Segmentation heads ──────────────────────────────────────────────
        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64,  64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_3 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

        # ── Fusion modules ──────────────────────────────────────────────────
        # ch_1 in each ICEF is increased by _H_CH because the height feature
        # map (B, _H_CH, H_s, W_s) is concatenated to the CNN feature before
        # the ICEF call.  All other dimensions stay the same.
        #
        #   Original  →  Updated
        #   ch_1=208  →  208 + _H_CH      (up_c,     deepest scale)
        #   ch_1=128  →  128 + _H_CH      (up_c_1_1, scale 3)
        #   ch_1= 48  →   48 + _H_CH      (up_c_2_1, scale 2)
        #   ch_1= 40  →   40 + _H_CH      (up_c_3_1, scale 1 / shallowest)
        self.up_c     = ICEF(ch_1=208 + _H_CH, ch_2=768, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_1_1 = ICEF(ch_1=128 + _H_CH, ch_2=384, ch_out=128, drop_rate=drop_rate / 2)
        self.up_c_1_2 = ACO(ch_1=256, ch_2=128, ch_out=128, if_PFN=True)
        self.up_c_2_1 = ICEF(ch_1=48  + _H_CH, ch_2=192, ch_out=64,  drop_rate=drop_rate / 2)
        self.up_c_2_2 = ACO(ch_1=128, ch_2=64,  ch_out=64,  if_PFN=False)
        self.up_c_3_1 = ICEF(ch_1=40  + _H_CH, ch_2=96,  ch_out=32,  drop_rate=drop_rate / 2)
        self.up_c_3_2 = ACO(ch_1=64,  ch_2=32,  ch_out=32,  if_PFN=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, imgs, building_height):
        """
        Args:
            imgs            : (B, 3, H, W)  — RGB image
            building_height : (B, 1, H, W)  — greyscale height raster.
                              Normalise to [0, 1] in the dataset before passing.
        Returns:
            map    : (B, num_classes, H, W)  segmentation logits
            mask_1 : (B, 1, H, W)            foreground mask at scale 3
            mask_2 : (B, 1, H, W)            foreground mask at scale 1
            bound1 : (B, 1, H, W)            boundary at scale 3
            bound2 : (B, 1, H, W)            boundary at scale 1
        """

        # ── Transformer path ────────────────────────────────────────────────
        x_b = self.transformer(imgs)

        x_b_1 = x_b[0]
        x_b_1 = torch.transpose(x_b_1, 1, 2).view(x_b_1.shape[0], -1, 64, 64)
        x_b_1 = self.drop(x_b_1)                               # (B,  96, 64, 64)

        x_b_2 = x_b[1]
        x_b_2 = torch.transpose(x_b_2, 1, 2).view(x_b_2.shape[0], -1, 32, 32)
        x_b_2 = self.drop(x_b_2)                               # (B, 192, 32, 32)

        x_b_3 = x_b[2]
        x_b_3 = torch.transpose(x_b_3, 1, 2).view(x_b_3.shape[0], -1, 16, 16)
        x_b_3 = self.drop(x_b_3)                               # (B, 384, 16, 16)

        x_b_4 = x_b[3]
        x_b_4 = torch.transpose(x_b_4, 1, 2).view(x_b_4.shape[0], -1, 8, 8)
        x_b_4 = self.drop(x_b_4)                               # (B, 768,  8,  8)

        # ── CNN path ────────────────────────────────────────────────────────
        x_u128 = self.efficienet.conv_stem(imgs)
        x_u128 = self.act1(self.efficienet.bn1(x_u128))
        x_u128 = self.efficienet.blocks[0](x_u128)
        x_u64  = self.efficienet.blocks[1](x_u128)             # (B,  40, 64, 64)

        x_u_2  = self.drop(self.efficienet.blocks[2](x_u64))  # (B,  48, 32, 32)

        x_u_3  = self.drop(self.efficienet.blocks[3](x_u_2))
        x_u_3  = self.drop(self.efficienet.blocks[4](x_u_3))  # (B, 128, 16, 16)

        x_u    = self.drop(self.efficienet.blocks[5](x_u_3))  # (B, 208,  8,  8)

        # ── Height encoder  ← NEW ───────────────────────────────────────────
        # building_height: (B, 1, H, W)  — greyscale, normalised [0, 1]
        # h1…h4 each have _H_CH=16 channels; spatial sizes mirror CNN stages.
        h1, h2, h3, h4 = self.height_encoder(building_height)
        #   h1: (B, 16, 64, 64)   h2: (B, 16, 32, 32)
        #   h3: (B, 16, 16, 16)   h4: (B, 16,  8,  8)

        # ── Inject height into CNN features before ICEF  ← NEW ─────────────
        # Concatenate along the channel dimension.  The ICEF ch_1 parameters
        # above already account for the extra _H_CH channels.
        x_u_h    = torch.cat([x_u,   h4], dim=1)  # (B, 208+16,  8,  8)
        x_u3_h   = torch.cat([x_u_3, h3], dim=1)  # (B, 128+16, 16, 16)
        x_u2_h   = torch.cat([x_u_2, h2], dim=1)  # (B,  48+16, 32, 32)
        x_u64_h  = torch.cat([x_u64, h1], dim=1)  # (B,  40+16, 64, 64)

        # ── Joint / decoder path ────────────────────────────────────────────
        x_c = self.up_c(x_u_h, x_b_4)                         # (B, 256,  8,  8)

        x_c_1_1 = self.up_c_1_1(x_u3_h, x_b_3)               # (B, 128, 16, 16)
        x_c_1, mask_1, bound1 = self.up_c_1_2(x_c, x_c_1_1)  # (B, 128, 16, 16)

        x_c_2_1 = self.up_c_2_1(x_u2_h, x_b_2)               # (B,  64, 32, 32)
        x_c_2   = self.up_c_2_2(x_c_1, x_c_2_1)              # (B,  64, 32, 32)

        x_c_3_1 = self.up_c_3_1(x_u64_h, x_b_1)              # (B,  32, 64, 64)
        x_c_3, mask_2, bound2 = self.up_c_3_2(x_c_2, x_c_3_1)

        # ── Segmentation heads ──────────────────────────────────────────────
        map_x_ = self.final_x(x_c)
        map_1_ = self.final_1(x_c_1)
        map_2_ = self.final_2(x_c_2)
        map_3_ = self.final_3(x_c_3)

        self.extract_features['fuse1'] = F.softmax(map_x_, dim=1)
        self.extract_features['fuse2'] = F.softmax(map_1_, dim=1)
        self.extract_features['fuse3'] = F.softmax(map_2_, dim=1)
        self.extract_features['fuse4'] = F.softmax(map_3_, dim=1)

        map_x = F.interpolate(map_x_, scale_factor=32, mode='bilinear')
        map_1 = F.interpolate(map_1_, scale_factor=16, mode='bilinear')
        map_2 = F.interpolate(map_2_, scale_factor=8,  mode='bilinear')
        map_3 = F.interpolate(map_3_, scale_factor=4,  mode='bilinear')
        map   = map_x + map_1 + map_2 + map_3

        self.extract_features['fuse'] = F.softmax(map, dim=1)

        mask_1 = F.interpolate(mask_1, scale_factor=16, mode='bilinear')
        mask_2 = F.interpolate(mask_2, scale_factor=4,  mode='bilinear')
        bound1 = F.interpolate(bound1, scale_factor=16, mode='bilinear')
        bound2 = F.interpolate(bound2, scale_factor=4,  mode='bilinear')

        return map, mask_1, mask_2, bound1, bound2

    # ─────────────────────────────────────────────────────────────────────────
    def init_weights(self):
        for module in [
            self.final_x, self.final_1, self.final_2, self.final_3,
            self.up_c, self.up_c_1_1, self.up_c_1_2,
            self.up_c_2_1, self.up_c_2_2,
            self.up_c_3_1, self.up_c_3_2,
            self.height_encoder,   # ← initialise height encoder
        ]:
            module.apply(init_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1,
                 bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                              padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn   = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x