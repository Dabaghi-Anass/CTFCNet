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

# -----------------------------
# Distributed Setup
# -----------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# -----------------------------
# Loss
# -----------------------------
def structure_loss(pred, mask):
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=6)
    wbce = ce_loss(pred, mask.squeeze(1).long())
    dice_loss = DiceLoss(6)
    dice = dice_loss(pred, mask, softmax=True)
    return wbce + dice

# -----------------------------
# Accuracy
# -----------------------------
def accuracy(pred, label):
    valid = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    return float(acc_sum) / (valid_sum + 1e-10)

# -----------------------------
# Train
# -----------------------------
def train(train_loader, model, optimizer, epoch, opt, writer, rank):
    model.train()
    loss_record = []
    acc_bank = []

    for i, (_, inputs, pack, mask, bound) in enumerate(tqdm(train_loader, disable=rank!=0)):

        images = inputs.cuda(rank, non_blocking=True).float()
        gts = pack.cuda(rank, non_blocking=True).float()
        masks = mask.cuda(rank, non_blocking=True).float()
        bounds = bound.cuda(rank, non_blocking=True).float()

        optimizer.zero_grad()

        map, bd2, bd1, bound2, bound1 = model(images)

        loss1 = structure_loss(map, gts)
        loss_bd2 = weighted_BCE_logits(bd2, masks)
        loss_bd1 = weighted_BCE_logits(bd1, masks)
        loss_bound2 = weighted_BCE_logits(bound2, bounds)
        loss_bound1 = weighted_BCE_logits(bound1, bounds)

        loss_2 = 0.6 * loss_bd1 + 0.4 * loss_bd2
        loss_3 = 0.6 * loss_bound1 + 0.4 * loss_bound2
        loss = 0.8 * loss1 + 0.1 * loss_2 + 0.1 * loss_3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()

        loss_record.append(loss.item())

        res = map.sigmoid()
        pred = torch.argmax(res, dim=1).cpu().numpy()
        gt = gts.cpu().numpy()
        acc_bank.append(accuracy(pred, gt))

    mean_loss = np.mean(loss_record)
    mean_acc = np.mean(acc_bank)

    if rank == 0:
        writer.add_scalar("train_loss", mean_loss, epoch)
        writer.add_scalar("train_acc", mean_acc, epoch)

    return mean_loss

# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(model, val_loader, rank):
    model.eval()
    loss_bank = []

    for _, inputs, pack, mask, bound in val_loader:

        images = inputs.cuda(rank).float()
        gts = pack.cuda(rank).float()

        res, _, _, _, _ = model(images)
        loss = structure_loss(res, gts)
        loss_bank.append(loss.item())

    return np.mean(loss_bank)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=2.0)
    parser.add_argument('--projectname', type=str, default="project")
    parser.add_argument('--data_name', type=str, default="dataset")
    parser.add_argument('--logs_path', type=str, default='./logs')
    parser.add_argument('--train_save', type=str, default='./checkpoints')
    opt = parser.parse_args()

    # Setup DDP
    rank = setup_ddp()

    cudnn.benchmark = True

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # Model
    model = CTCFNet(pretrained=True).cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

    # Dataset
    train_dataset = data_pre.Data('train', random_flip=True)
    val_dataset = data_pre.Data('test')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchsize,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batchsize,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Logging only on rank 0
    writer = None
    if rank == 0:
        writer = SummaryWriter(os.path.join(opt.logs_path, opt.projectname, opt.data_name))

    best_loss = 1e9

    for epoch in range(opt.epoch):
        train_sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, optimizer, epoch, opt, writer, rank)
        val_loss = validate(model, val_loader, rank)

        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

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