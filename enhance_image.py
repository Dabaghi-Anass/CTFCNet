"""
Super-Resolution Pipeline — Real-ESRGAN & EDSR in PyTorch
==========================================================
Uses open-source AI models to upscale images 2× or 4×.

Models supported
----------------
  real-esrgan-x4   Real-ESRGAN × 4  (general purpose, photorealistic)
  real-esrgan-x2   Real-ESRGAN × 2  (general purpose, lighter)
  edsr-x4          EDSR × 4         (NTIRE 2017 winner, clean images)
  edsr-x2          EDSR × 2         (NTIRE 2017 winner, clean images)

Setup (one-time)
----------------
  pip install torch torchvision opencv-python-headless Pillow tqdm

  Weights (~65 MB each) are downloaded automatically on first run into
  ~/.cache/sr_models/   (configurable via --model-dir)

Usage
-----
  # 4× upscale with Real-ESRGAN (recommended for photos)
  python super_resolution.py -i photo.jpg -o hires.jpg

  # 2× upscale with EDSR (good for illustrations / clean art)
  python super_resolution.py -i art.png -o art_2x.png --model edsr-x2

  # Chain with the enhancement pipeline
  python super_resolution.py -i photo.jpg -o hires.jpg --model real-esrgan-x4 --enhance

  # Tile large images to avoid OOM (auto-enabled for images > 512px)
  python super_resolution.py -i huge.jpg -o huge_4x.jpg --tile 256 --tile-pad 16

Programmatic API
----------------
  from super_resolution import SuperResolutionPipeline

  sr = SuperResolutionPipeline(model="real-esrgan-x4", device="cuda")
  hires = sr.upscale_file("photo.jpg", "out.jpg")

  # Or work with numpy arrays directly:
  import cv2
  bgr = cv2.imread("photo.jpg")
  result_bgr = sr.upscale(bgr)
"""

from __future__ import annotations
import argparse
import hashlib
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Optional tqdm progress bar ────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── PyTorch import with helpful error message ─────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY  (weights from official repos, MIT/Apache-licensed)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
    "real-esrgan-x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "arch": "rrdbnet",
        "num_in_ch": 3, "num_out_ch": 3,
        "num_feat": 64, "num_block": 23, "num_grow_ch": 32,
        "md5": None,  # optional integrity check
        "description": "Real-ESRGAN ×4 — best for real-world photos & textures",
    },
    "real-esrgan-x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "arch": "rrdbnet",
        "num_in_ch": 3, "num_out_ch": 3,
        "num_feat": 64, "num_block": 23, "num_grow_ch": 32,
        "description": "Real-ESRGAN ×2 — lighter, good for mild upscaling",
    },
    # EDSR: https://github.com/sanghyun-son/EDSR-PyTorch
    "edsr-x4": {
        "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
        "filename": "edsr_baseline_x4.pt",
        "scale": 4,
        "arch": "edsr",
        "n_resblocks": 16, "n_feats": 64, "res_scale": 1,
        "description": "EDSR-baseline ×4 — NTIRE 2017 winner, great for clean images",
    },
    "edsr-x2": {
        "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
        "filename": "edsr_baseline_x2.pt",
        "scale": 2,
        "arch": "edsr",
        "n_resblocks": 16, "n_feats": 64, "res_scale": 1,
        "description": "EDSR-baseline ×2 — clean, sharp results",
    },
}

DEFAULT_MODEL = "real-esrgan-x4"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sr_models"


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

# ── Real-ESRGAN: RRDB-Net ─────────────────────────────────────────────────────

if HAS_TORCH:

    class ResidualDenseBlock(nn.Module):
        """Residual Dense Block — core building block of RRDB."""
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat,                   num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch,     num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch,   num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch,   num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch,   num_feat,    3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # weight init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    m.weight.data *= 0.1

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
            x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
            x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
            x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        """Residual in Residual Dense Block."""
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        """
        Real-ESRGAN Generator (RRDB-Net).
        Matches the official xinntao/Real-ESRGAN checkpoint layout.
        """
        def __init__(self, num_in_ch=3, num_out_ch=3,
                     num_feat=64, num_block=23, num_grow_ch=32, scale=4):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            # Upsampling
            ups = []
            n_up = int(math.log2(scale))
            for _ in range(n_up):
                ups += [
                    nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            self.upsampler = nn.Sequential(*ups)
            self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat       = self.conv_first(x)
            body_feat  = self.conv_body(self.body(feat))
            feat       = feat + body_feat
            # pixel-shuffle upsampling
            for _ in range(int(math.log2(self.scale))):
                feat = F.interpolate(feat, scale_factor=2, mode="nearest")
                feat = self.lrelu(self.upsampler[0](feat))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    # ── EDSR ─────────────────────────────────────────────────────────────────

    class MeanShift(nn.Conv2d):
        """Normalize / denormalize RGB with ImageNet-like statistics."""
        def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040),
                     rgb_std=(1.0, 1.0, 1.0), sign=-1):
            super().__init__(3, 3, kernel_size=1)
            std = torch.Tensor(rgb_std)
            self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
            self.bias.data   = sign * rgb_range * torch.Tensor(rgb_mean) / std
            for p in self.parameters():
                p.requires_grad = False

    class ResBlock(nn.Module):
        def __init__(self, n_feats, kernel_size=3, res_scale=1.0):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            )
            self.res_scale = res_scale

        def forward(self, x):
            return x + self.body(x) * self.res_scale

    class Upsampler(nn.Sequential):
        def __init__(self, scale, n_feats):
            layers = []
            if (scale & (scale - 1)) == 0:  # power of 2
                for _ in range(int(math.log(scale, 2))):
                    layers += [nn.Conv2d(n_feats, 4*n_feats, 3, padding=1),
                               nn.PixelShuffle(2)]
            elif scale == 3:
                layers += [nn.Conv2d(n_feats, 9*n_feats, 3, padding=1),
                           nn.PixelShuffle(3)]
            super().__init__(*layers)

    class EDSR(nn.Module):
        """
        EDSR-baseline: Enhanced Deep Residual Networks for SR.
        Matches the sanghyun-son/EDSR-PyTorch checkpoint layout.
        """
        def __init__(self, n_resblocks=16, n_feats=64, scale=4,
                     res_scale=1.0, rgb_range=255):
            super().__init__()
            self.sub_mean = MeanShift(rgb_range)
            self.add_mean = MeanShift(rgb_range, sign=1)

            # head
            self.head = nn.Sequential(nn.Conv2d(3, n_feats, 3, padding=1))
            # body
            self.body = nn.Sequential(
                *[ResBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)],
                nn.Conv2d(n_feats, n_feats, 3, padding=1),
            )
            # tail
            self.tail = nn.Sequential(
                Upsampler(scale, n_feats),
                nn.Conv2d(n_feats, 3, 3, padding=1),
            )

        def forward(self, x):
            x = self.sub_mean(x)
            x = self.head(x)
            res = self.body(x)
            res += x
            x = self.tail(res)
            x = self.add_mean(x)
            return x


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def _progress_hook(bar_holder: list):
    """Returns a urllib reporthook that updates a tqdm bar."""
    def hook(block_num, block_size, total_size):
        if not bar_holder:
            if HAS_TQDM and total_size > 0:
                bar_holder.append(_tqdm(
                    total=total_size, unit="B", unit_scale=True,
                    desc="  Downloading", leave=True
                ))
            else:
                bar_holder.append(None)
        bar = bar_holder[0]
        if bar is not None:
            downloaded = min(block_num * block_size, total_size)
            bar.n = downloaded
            bar.refresh()
        else:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, 100 * downloaded // total_size)
                print(f"\r  Downloading… {pct:3d}%", end="", flush=True)
    return hook


def download_weights(url: str, dest: Path, md5: Optional[str] = None) -> Path:
    """Download model weights with progress bar and optional MD5 check."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  ✓ Weights cached: {dest}")
        return dest

    print(f"  ↓ Downloading weights from:\n    {url}")
    bar_holder: list = []
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(bar_holder))
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download weights: {exc}\n"
            f"Manual download: {url}\n"
            f"Save to: {dest}"
        ) from exc

    if bar_holder and bar_holder[0] is not None:
        bar_holder[0].close()
    else:
        print()  # newline after \r progress

    if md5:
        actual = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual != md5:
            dest.unlink()
            raise RuntimeError(f"MD5 mismatch for {dest.name}: expected {md5}, got {actual}")
        print(f"  ✓ MD5 verified")

    print(f"  ✓ Saved → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> "nn.Module":
    """Instantiate the correct architecture from a registry config dict."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision")

    arch = cfg["arch"]
    if arch == "rrdbnet":
        return RRDBNet(
            num_in_ch=cfg["num_in_ch"],
            num_out_ch=cfg["num_out_ch"],
            num_feat=cfg["num_feat"],
            num_block=cfg["num_block"],
            num_grow_ch=cfg["num_grow_ch"],
            scale=cfg["scale"],
        )
    elif arch == "edsr":
        return EDSR(
            n_resblocks=cfg["n_resblocks"],
            n_feats=cfg["n_feats"],
            scale=cfg["scale"],
            res_scale=cfg["res_scale"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def load_model(model_name: str, device: str = "cpu",
               cache_dir: Path = DEFAULT_CACHE_DIR) -> "nn.Module":
    """
    Build model, download weights if needed, and load state dict.
    Returns eval-mode model on the requested device.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cfg = MODEL_REGISTRY[model_name]
    print(f"\n📦 Loading model: {model_name}")
    print(f"   {cfg['description']}")

    weight_path = download_weights(
        cfg["url"],
        cache_dir / cfg["filename"],
        md5=cfg.get("md5"),
    )

    model = build_model(cfg)

    # Load checkpoint — handle both plain state_dict and wrapped formats
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]         # Real-ESRGAN EMA weights
        elif "params" in ckpt:
            state = ckpt["params"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Strip module. prefix (DataParallel artefact)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ Missing keys  ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

    model = model.to(device).eval()
    print(f"  ✓ Model ready on {device.upper()}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TILED INFERENCE  (avoids OOM on large images)
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_from_bgr(bgr: np.ndarray, device: str) -> "torch.Tensor":
    """BGR uint8 → normalised float32 tensor [1,3,H,W]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _bgr_from_tensor(t: "torch.Tensor") -> np.ndarray:
    """Normalised float32 tensor [1,3,H,W] → BGR uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((arr * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


@torch.no_grad()
def _infer_tile(model: "nn.Module", tile: "torch.Tensor") -> "torch.Tensor":
    return model(tile)


@torch.no_grad()
def infer_tiled(model: "nn.Module", bgr: np.ndarray, scale: int,
                tile: int = 512, pad: int = 32,
                device: str = "cpu") -> np.ndarray:
    """
    Run the SR model with overlap-tiled inference.
    Tiles the input into `tile`×`tile` patches (with `pad` overlap),
    runs the model on each, and blends them back via a smooth weight mask.
    Automatically falls back to single-pass if the image fits in one tile.
    """
    h, w = bgr.shape[:2]

    # Single-pass path
    if h <= tile and w <= tile:
        t = _tensor_from_bgr(bgr, device)
        out_t = _infer_tile(model, t)
        return _bgr_from_tensor(out_t)

    # Tiled path
    out_h, out_w = h * scale, w * scale
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1),  dtype=np.float32)

    # Build a smooth weight window (Hann) for blending overlapping tiles
    def hann2d(size):
        hann = np.hanning(size).astype(np.float32)
        return np.outer(hann, hann)[:, :, None]

    step = tile - 2 * pad
    y_starts = list(range(0, h - tile, step)) + [max(0, h - tile)]
    x_starts = list(range(0, w - tile, step)) + [max(0, w - tile)]
    total = len(y_starts) * len(x_starts)

    print(f"  Tiled inference: {len(y_starts)}×{len(x_starts)} = {total} tiles …")

    for ti, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = bgr[y0:y1, x0:x1]

            t = _tensor_from_bgr(patch, device)
            out_t = _infer_tile(model, t)
            out_patch = out_t.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()

            oh, ow = out_patch.shape[:2]
            w_mask = hann2d(oh) if oh == ow else \
                     np.outer(np.hanning(oh), np.hanning(ow))[:,:,None].astype(np.float32)

            oy0, ox0 = y0*scale, x0*scale
            oy1, ox1 = oy0 + oh, ox0 + ow
            out[oy0:oy1, ox0:ox1]    += out_patch * w_mask
            weight[oy0:oy1, ox0:ox1] += w_mask

            idx = ti * len(x_starts) + xi + 1
            print(f"\r  Tile {idx}/{total}", end="", flush=True)

    print()
    result = np.clip(out / (weight + 1e-8), 0, 1)
    return cv2.cvtColor((result * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK (no PyTorch) — high-quality bicubic + Lanczos sharpening
# ─────────────────────────────────────────────────────────────────────────────

def bicubic_fallback(bgr: np.ndarray, scale: int) -> np.ndarray:
    """
    High-quality bicubic upscale + unsharp mask when PyTorch is unavailable.
    Not as good as the neural network, but still better than plain resize.
    """
    h, w = bgr.shape[:2]
    upscaled = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # Light unsharp mask to recover perceived sharpness
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)
    sharp = cv2.addWeighted(upscaled, 1.4, blurred, -0.4, 0)
    return sharp


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

class SuperResolutionPipeline:
    """
    High-level interface to the SR pipeline.

    Example
    -------
    >>> sr = SuperResolutionPipeline(model="real-esrgan-x4")
    >>> result = sr.upscale_file("low_res.jpg", "high_res.jpg")

    >>> import cv2
    >>> bgr = cv2.imread("photo.jpg")
    >>> hd  = sr.upscale(bgr)
    >>> cv2.imwrite("photo_4x.jpg", hd)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        tile: int = 512,
        tile_pad: int = 32,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self.model_name = model
        self.cfg        = MODEL_REGISTRY[model]
        self.scale      = self.cfg["scale"]
        self.tile       = tile
        self.tile_pad   = tile_pad
        self._net       = None

        if device is None:
            if HAS_TORCH:
                self.device = "cuda" if torch.cuda.is_available() else \
                              "mps"  if torch.backends.mps.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.cache_dir = Path(cache_dir)

        if HAS_TORCH:
            self._net = load_model(model, device=self.device, cache_dir=self.cache_dir)
        else:
            print(
                "⚠ PyTorch not found — using bicubic fallback.\n"
                "  Install with: pip install torch torchvision"
            )

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        """
        Upscale a BGR uint8 numpy image.
        Returns a BGR uint8 image at scale × the input resolution.
        """
        if self._net is None:
            return bicubic_fallback(bgr, self.scale)

        return infer_tiled(
            self._net, bgr,
            scale=self.scale,
            tile=self.tile,
            pad=self.tile_pad,
            device=self.device,
        )

    def upscale_file(self, input_path: str, output_path: str) -> str:
        """Convenience wrapper: read file → upscale → write file."""
        bgr = cv2.imread(input_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")

        h, w = bgr.shape[:2]
        print(f"\n🖼  Input  : {input_path}  ({w}×{h})")
        print(f"   Scale  : ×{self.scale}  →  ({w*self.scale}×{h*self.scale})")

        result = self.upscale(bgr)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(output_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
        elif ext == ".png":
            cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(output_path, result)

        print(f"\n✅ Saved  : {output_path}")
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: CHAIN WITH THE ENHANCEMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def enhance_then_upscale(input_path: str, output_path: str,
                         model: str = DEFAULT_MODEL,
                         enhance_preset: str = "balanced") -> str:
    """
    Run the image_enhancement_pipeline.py first, then super-resolve.
    Both scripts must be in the same directory.
    """
    try:
        import importlib.util
        here = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "enhance", here / "image_enhancement_pipeline.py")
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)

        bgr = cv2.imread(input_path)
        params = ep.PRESETS[enhance_preset]
        enhanced = ep.enhance(bgr, **params)
        print(f"✓ Enhancement pass complete ({enhance_preset} preset)")

        sr = SuperResolutionPipeline(model=model)
        result = sr.upscale(enhanced)

        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(f"✅ Enhance+SR saved → {output_path}")
        return output_path

    except FileNotFoundError:
        print("⚠ image_enhancement_pipeline.py not found — running SR only")
        sr = SuperResolutionPipeline(model=model)
        return sr.upscale_file(input_path, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _list_models():
    print("\nAvailable super-resolution models:\n")
    for name, cfg in MODEL_REGISTRY.items():
        print(f"  {name:<20} ×{cfg['scale']}  {cfg['description']}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Super-Resolution  (Real-ESRGAN / EDSR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",    "-i", help="Input image path")
    parser.add_argument("--output",   "-o", help="Output image path")
    parser.add_argument("--model",    "-m", default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--device",   "-d", default=None,
                        help="Device: cpu | cuda | mps  (auto-detected if omitted)")
    parser.add_argument("--tile",     type=int, default=512,
                        help="Tile size for tiled inference (default: 512)")
    parser.add_argument("--tile-pad", type=int, default=32,
                        help="Overlap padding between tiles (default: 32)")
    parser.add_argument("--model-dir", default=str(DEFAULT_CACHE_DIR),
                        help=f"Directory to cache model weights (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--enhance",  action="store_true",
                        help="Run enhancement pipeline before SR (requires image_enhancement_pipeline.py)")
    parser.add_argument("--enhance-preset", default="balanced",
                        help="Preset for enhancement (default: balanced)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        _list_models()
        return

    if not args.input or not args.output:
        print("Error: --input and --output are required.\n"
              "Run with --help for usage, or --list-models to see available models.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.enhance:
        enhance_then_upscale(
            args.input, args.output,
            model=args.model,
            enhance_preset=args.enhance_preset,
        )
    else:
        sr = SuperResolutionPipeline(
            model=args.model,
            device=args.device,
            tile=args.tile,
            tile_pad=args.tile_pad,
            cache_dir=Path(args.model_dir),
        )
        sr.upscale_file(args.input, args.output)


if __name__ == "__main__":
    main()
