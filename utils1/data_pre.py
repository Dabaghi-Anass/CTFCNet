import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import cv2


num_classes = 6
ST_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
ST_CLASS = ['nonbd', 'complex bd', "simple bd", "regular bd", "irregular bd", "large-scale bd"]

# UBTD
MEAN_A = np.array([62.239872, 63.211044, 66.932594])
STD_A  = np.array([46.32426,  44.79888,  44.923523])

root = '/kaggle/input/datasets/anassdabaghi/casablanca-dataset/dataset'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    return [Color2Index(d) for d in ColorLabels]


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    return colormap[np.asarray(pred, dtype='int32'), :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    return im


def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def normalize_height(h: np.ndarray) -> np.ndarray:
    """
    Normalise a greyscale height raster to [0, 1].
    Handles edge-case of a flat (constant) raster gracefully.
    h : (H, W) float32 array
    """
    h = h.astype(np.float32)
    h_min, h_max = h.min(), h.max()
    if h_max - h_min > 1e-6:
        h = (h - h_min) / (h_max - h_min)
    else:
        h = np.zeros_like(h)
    return h


# ─────────────────────────────────────────────────────────────────────────────
# read_RSimages — now also loads height maps from heights1/
# ─────────────────────────────────────────────────────────────────────────────

def read_RSimages(mode, rescale=False):
    img_A_dir       = os.path.join(root, mode, 'img1')
    label_A_dir     = os.path.join(root, mode, 'label1')
    label_bound_dir = os.path.join(root, mode, 'boundary1')
    height_dir      = os.path.join(root, mode, 'heights1')   # ← NEW

    data_list = os.listdir(img_A_dir)

    imgs_list_A  = []
    labels_A     = []
    label_mask   = []
    label_bound  = []
    heights_list = []   # ← NEW

    count = 0
    for it in data_list:
        if it[-4:] == '.png':
            img_A_path    = os.path.join(img_A_dir,       it)
            label_A_path  = os.path.join(label_A_dir,     it)
            bound_path    = os.path.join(label_bound_dir, it)
            height_path   = os.path.join(height_dir,      it)   # ← NEW

            # ── RGB image path (loaded lazily in __getitem__) ────────────────
            imgs_list_A.append(img_A_path)

            # ── Semantic label ───────────────────────────────────────────────
            label_A = io.imread(label_A_path)
            label_A = np.nan_to_num(label_A)

            # ── Foreground mask ──────────────────────────────────────────────
            label_A_mask = np.zeros_like(label_A)
            label_A_mask[label_A != 0] = 1

            # ── Boundary mask ────────────────────────────────────────────────
            label_bound1 = io.imread(bound_path)
            label_bound1 = np.nan_to_num(label_bound1)
            label_bound_mask = np.zeros_like(label_bound1)
            label_bound_mask[label_bound1 != 0] = 1

            # ── Height raster  ← NEW ─────────────────────────────────────────
            # Load as greyscale (single channel).  PNG bit-depth is handled
            # automatically by skimage; we cast to float32 then normalise.
            height_img = io.imread(height_path, as_gray=True)   # (H, W) float64
            height_img = np.nan_to_num(height_img)
            height_img = normalize_height(height_img)           # [0, 1] float32

            labels_A.append(label_A)
            label_mask.append(label_A_mask)
            label_bound.append(label_bound_mask)
            heights_list.append(height_img)                     # ← NEW

        count += 1
        if not count % 500:
            print(f'{count}/{len(data_list)} images loaded.')

    print(labels_A[0].shape)
    print(f'{len(imgs_list_A)} {mode} images loaded.')

    return imgs_list_A, labels_A, label_mask, label_bound, heights_list   # ← NEW


# ─────────────────────────────────────────────────────────────────────────────
# Data — main dataset class
# __getitem__ now returns a 6-tuple:
#   (name, image, label, fg_mask, boundary_mask, height)
# height shape: (1, H, W), dtype float32, range [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        (self.imgs_list_A,
         self.labels_A,
         self.label_mask,
         self.label_bound,
         self.heights)      = read_RSimages(mode)   # ← unpack 5-tuple

    def get_mask_name(self, idx):
        return os.path.split(self.imgs_list_A[idx])[-1]

    def __getitem__(self, idx):
        # ── RGB image ────────────────────────────────────────────────────────
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = np.nan_to_num(img_A)
        img_A = normalize_image(img_A, 'A')

        # ── Labels / masks ───────────────────────────────────────────────────
        label_A          = self.labels_A[idx]
        label_mask       = self.label_mask[idx]
        label_bound_mask = self.label_bound[idx]

        # ── Height raster  ← NEW ─────────────────────────────────────────────
        # self.heights[idx] is (H, W) float32 in [0, 1].
        # F.to_tensor expects HxW or HxWxC numpy; for a 2-D array we add a
        # channel dim first so the output is (1, H, W).
        height = self.heights[idx]                              # (H, W)
        height_tensor = torch.from_numpy(height).unsqueeze(0)  # (1, H, W)

        return (
            self.get_mask_name(idx),          # str
            F.to_tensor(img_A),               # (3, H, W) float32
            torch.from_numpy(label_A),        # (H, W) int / uint
            torch.from_numpy(label_mask),     # (H, W) uint8
            torch.from_numpy(label_bound_mask),  # (H, W) uint8
            height_tensor,                    # (1, H, W) float32  ← NEW
        )

    def __len__(self):
        return len(self.imgs_list_A)


# ─────────────────────────────────────────────────────────────────────────────
# Data_test — inference-only dataset (no labels, no height needed for output)
# Height is still loaded so the model can run end-to-end during test.
# ─────────────────────────────────────────────────────────────────────────────

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A          = []
        self.mask_name_list  = []
        self.heights         = []   # ← NEW

        imgA_dir   = os.path.join(test_dir, 'img1')
        height_dir = os.path.join(test_dir, 'heights1')   # ← NEW
        data_list  = os.listdir(imgA_dir)

        for it in data_list:
            if it[-4:] == '.png':
                img_path    = os.path.join(imgA_dir,   it)
                height_path = os.path.join(height_dir, it)   # ← NEW

                self.imgs_A.append(io.imread(img_path))
                self.mask_name_list.append(it)

                # ── Height ───────────────────────────────────────────────────
                height_img = io.imread(height_path, as_gray=True)
                height_img = np.nan_to_num(height_img)
                height_img = normalize_height(height_img)
                self.heights.append(height_img)              # ← NEW

        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = normalize_image(self.imgs_A[idx], 'A')

        height = self.heights[idx]                              # (H, W)
        height_tensor = torch.from_numpy(height).unsqueeze(0)  # (1, H, W)

        return (
            self.get_mask_name(idx),   # str
            F.to_tensor(img_A),        # (3, H, W)
            height_tensor,             # (1, H, W) float32  ← NEW
        )

    def __len__(self):
        return self.len