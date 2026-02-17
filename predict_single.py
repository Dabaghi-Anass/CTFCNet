import os
import numpy as np
import torch
import argparse
from skimage import io
from utils1 import data_pre
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from models.yynet_efficient_swin import CTCFNet as net
from torchvision import transforms
from PIL import Image
import time


class PredOption():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--image_path', required=False, default='./', help='Path to the input image')
        parser.add_argument('--pred_dir',   required=False, default='./', help='Location to save the prediction result')
        parser.add_argument('--chkpt_path', required=False, default='./', help='Path to model checkpoint')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_known_args()[0]

    def parser(self):
        self.opt = self.gather_options()
        return self.opt


def load_image(image_path, img_size=256):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension: (1, 3, H, W)
    return img_tensor


def predict_single(model, image_path, pred_dir, device):
    os.makedirs(pred_dir, exist_ok=True)

    # Load and preprocess image
    imgs = load_image(image_path).to(device).float()

    # Inference
    with torch.no_grad():
        preds, _, _, _, _ = model(imgs)

    preds = preds.cpu().detach()
    preds = nn.Softmax(dim=1)(preds)
    preds = preds.argmax(dim=1).long().numpy().squeeze()  # shape: (H, W)

    img_filename = os.path.basename(image_path)
    pred_save_path = os.path.join(pred_dir, img_filename)
    io.imsave(pred_save_path, data_pre.Index2Color(preds))
    print(f"Prediction saved to: {pred_save_path}")


def main():
    opt = PredOption().parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    model = net(pretrained=True)
    model.load_state_dict(torch.load(opt.chkpt_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    start_time = time.time()
    predict_single(model, opt.image_path, opt.pred_dir, device)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()