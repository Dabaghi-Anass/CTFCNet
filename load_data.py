import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn

from utils1.utils import *
from utils1 import data_pre


cudnn.benchmark = True

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

train_dataset = data_pre.Data('train', random_flip=True)
val_dataset = data_pre.Data('test')

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    sampler=val_sampler,
    num_workers=2,
    pin_memory=True
)


