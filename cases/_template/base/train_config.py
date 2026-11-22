"""Training configuration (agents may edit). Keep schedules relative to total_steps."""
import torch

LR = 3e-3
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0


def loss_fn(pred, y):
    return torch.nn.functional.mse_loss(pred, y)


def make_optimizer(params):
    return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)


def make_scheduler(opt, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
