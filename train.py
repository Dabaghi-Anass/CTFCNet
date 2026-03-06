import os
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn

from models.yynet_efficient_swin import CTCFNet
from utils1.utils import *
from utils1 import data_pre
from tensorboardX import SummaryWriter


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def structure_loss(pred, mask):
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=6)
    wbce    = ce_loss(pred, mask.squeeze(1).long())
    dice    = DiceLoss(6)(pred, mask, softmax=True)
    return wbce + dice


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(pred, label):
    valid   = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    return float(acc_sum) / (valid.sum() + 1e-10)

def miou(pred, label, num_classes=7):
    iou_list = []
    for i in range(num_classes):
        intersection = ((pred == i) & (label == i)).sum()
        union        = ((pred == i) | (label == i)).sum()
        iou_list.append(0 if union == 0 else intersection / union)
    return np.mean(iou_list)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, epoch, opt, writer, rank):
    model.train()
    loss_record, acc_bank, miou_bank = [], [], []

    # ── Dataloader now yields building_height as the 6th item ────────────────
    # Expected tuple from data_pre.Data:
    #   (name, image, label_map, foreground_mask, boundary_mask, height_map)
    #
    #   height_map : (B, 1, H, W)  greyscale raster, normalised [0, 1]
    for i, (_, inputs, pack, mask, bound, height) in enumerate(
            tqdm(train_loader, disable=rank != 0)):

        images  = inputs.cuda(rank, non_blocking=True).float()
        gts     = pack.cuda(rank,   non_blocking=True).float()
        masks   = mask.cuda(rank,   non_blocking=True).float()
        bounds  = bound.cuda(rank,  non_blocking=True).float()
        heights = height.cuda(rank, non_blocking=True).float()   # ← NEW

        optimizer.zero_grad()

        # ── Forward — pass building_height to the model  ← NEW ───────────────
        map_out, bd2, bd1, bound2, bound1 = model(images, heights)

        loss1       = structure_loss(map_out, gts)
        loss_bd2    = weighted_BCE_logits(bd2,    masks)
        loss_bd1    = weighted_BCE_logits(bd1,    masks)
        loss_bound2 = weighted_BCE_logits(bound2, bounds)
        loss_bound1 = weighted_BCE_logits(bound1, bounds)

        loss_2 = 0.6 * loss_bd1 + 0.4 * loss_bd2
        loss_3 = 0.6 * loss_bound1 + 0.4 * loss_bound2
        loss   = 0.8 * loss1 + 0.1 * loss_2 + 0.1 * loss_3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()

        loss_record.append(loss.item())

        res  = map_out.sigmoid()
        pred = torch.argmax(res, dim=1).cpu().numpy()
        gt   = gts.cpu().numpy()
        acc_bank.append(accuracy(pred, gt))
        miou_bank.append(miou(pred, gt))

    mean_loss = np.mean(loss_record)
    mean_acc  = np.mean(acc_bank)
    mean_miou = np.mean(miou_bank)

    if rank == 0:
        writer.add_scalar("train_loss", mean_loss, epoch)
        writer.add_scalar("train_acc",  mean_acc,  epoch)
        writer.add_scalar("train_miou", mean_miou, epoch)

    return mean_loss


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, rank):
    model.eval()
    loss_bank, acc_bank, miou_bank = [], [], []

    # ── Unpack height from the validation loader  ← NEW ─────────────────────
    for _, inputs, pack, mask, bound, height in val_loader:

        images  = inputs.cuda(rank).float()
        gts     = pack.cuda(rank).float()
        heights = height.cuda(rank).float()   # ← NEW

        # ── Forward — pass building_height  ← NEW ────────────────────────────
        res, _, _, _, _ = model(images, heights)

        loss = structure_loss(res, gts)
        loss_bank.append(loss.item())

        pred = torch.argmax(res.sigmoid(), dim=1).cpu().numpy()
        gt   = gts.cpu().numpy()
        acc_bank.append(accuracy(pred, gt))
        miou_bank.append(miou(pred, gt))

    return np.mean(loss_bank), np.mean(acc_bank), np.mean(miou_bank)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=120)
    parser.add_argument('--batchsize',   type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--grad_norm',   type=float, default=2.0)
    parser.add_argument('--projectname', type=str,   default="project")
    parser.add_argument('--data_name',   type=str,   default="dataset")
    parser.add_argument('--logs_path',   type=str,   default='./logs')
    parser.add_argument('--train_save',  type=str,   default='./checkpoints')
    parser.add_argument('--pretrained',  action='store_true',
                        help='Use pretrained backbone weights')
    parser.add_argument('--checkpoint',  type=str,   default=None,
                        help='Path to a checkpoint to resume from')
    opt = parser.parse_args()

    # ── DDP ─────────────────────────────────────────────────────────────────
    rank = setup_ddp()
    cudnn.benchmark = True

    random.seed(1234);      np.random.seed(1234)
    torch.manual_seed(1234); torch.cuda.manual_seed(1234)

    # ── Model ────────────────────────────────────────────────────────────────
    model = CTCFNet(pretrained=opt.pretrained).cuda(rank)
    if rank == 0:
        print("Model initialised",
              "WITH pretrained backbone" if opt.pretrained else "WITHOUT pretrained backbone")

    # ── Optional checkpoint resume ───────────────────────────────────────────
    if opt.checkpoint is not None and os.path.exists(opt.checkpoint):
        if rank == 0:
            print(f"Loading checkpoint: {opt.checkpoint}")
        ckpt = torch.load(opt.checkpoint, map_location=f'cuda:{rank}')

        model_keys      = set(model.state_dict().keys())
        ckpt_filtered   = {k: v for k, v in ckpt.items() if k in model_keys}

        if rank == 0:
            skipped = set(ckpt.keys()) - set(ckpt_filtered.keys())
            print(f"  Loaded {len(ckpt_filtered)} keys, skipped {len(skipped)}")
            # Note: height_encoder weights will NOT be in old checkpoints —
            # they initialise from scratch automatically (strict=False below).

        model.load_state_dict(ckpt_filtered, strict=False)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

    # ── Dataset ──────────────────────────────────────────────────────────────
    # IMPORTANT: data_pre.Data must be updated to load and return the height
    # raster as a 6th tensor.  Expected __getitem__ return signature:
    #
    #   return name, image, label, foreground_mask, boundary_mask, height
    #
    # where `height` is a (1, H, W) float tensor normalised to [0, 1].
    # A minimal example for loading from a greyscale PNG:
    #
    #   from PIL import Image
    #   import torchvision.transforms.functional as TF
    #
    #   height_img = Image.open(height_path).convert('L')   # greyscale
    #   height_img = TF.resize(height_img, [H, W])
    #   height = TF.to_tensor(height_img)                   # [0, 1], shape (1,H,W)
    #   # Optional: normalise to zero-mean/unit-variance if values vary a lot
    #   # height = (height - height.mean()) / (height.std() + 1e-6)
    train_dataset = data_pre.Data('train', random_flip=True)
    val_dataset   = data_pre.Data('test')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batchsize,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batchsize,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )

    # ── TensorBoard (rank-0 only) ────────────────────────────────────────────
    writer = None
    if rank == 0:
        writer = SummaryWriter(
            os.path.join(opt.logs_path, opt.projectname, opt.data_name)
        )

    best_loss = 1e9

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(opt.epoch):
        train_sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, optimizer, epoch, opt, writer, rank)
        val_loss, val_acc, val_miou = validate(model, val_loader, rank)
        scheduler.step()

        if rank == 0:
            writer.add_scalar("val_loss", val_loss, epoch)
            writer.add_scalar("val_acc",  val_acc,  epoch)
            writer.add_scalar("val_miou", val_miou, epoch)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} | "
                  f"Val Loss {val_loss:.4f} | "
                  f"Val Acc {val_acc:.4f} | "
                  f"Val mIoU {val_miou:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs(opt.train_save, exist_ok=True)
                torch.save(
                    model.module.state_dict(),
                    os.path.join(opt.train_save, "best_model.pth")
                )

    if rank == 0:
        writer.close()

    cleanup_ddp()