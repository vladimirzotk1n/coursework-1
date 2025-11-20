import torch
import torch.nn as nn
from .metrics import Metrics

metrics_scorer = Metrics()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        dice_per_class = metrics_scorer.dice(preds, labels)
        return 1.0 - torch.mean(dice_per_class)


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, preds, labels):
        dice_loss = self.dice_loss(preds, labels)
        ce_loss = self.ce(preds, torch.argmax(labels, dim=1))
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss