import pytest
import torch
import torch.nn.functional as F
from coursework_1.modeling.metrics import Metrics

def test_metrics():
    metrics = Metrics()
    B, C, H, W = 16, 4, 128, 128

    preds = torch.rand(B, C, H, W)
    labels_idx = torch.randint(0, C, (B, H, W))
    labels = F.one_hot(labels_idx, num_classes=C).permute(0, 3, 1, 2).float()

    dice_per_class = metrics.dice(preds, labels, per_class=True)
    dice_mean = metrics.dice(preds, labels, per_class=False)

    assert dice_per_class.shape[0] == C, "Dice per class имеет неправильную форму"
    assert 0 <= dice_mean <= 1, f"Dice mean вне диапазона [0, 1]: {dice_mean}"
    assert torch.all((dice_per_class >= 0) & (dice_per_class <= 1)), "Dice per class вне диапазона [0, 1]"


    iou_per_class = metrics.iou(preds, labels, per_class=True)
    iou_mean = metrics.iou(preds, labels, per_class=False)

    assert iou_per_class.shape[0] == C, "IoU per class имеет неправильную форму"
    assert 0 <= iou_mean <= 1, f"IoU mean вне диапазона [0, 1]"
    assert torch.all((iou_per_class >= 0) & (iou_per_class <= 1)), "IoU per class вне диапазона [0, 1]"

    class_metrics = metrics.classification_metrics(preds, labels)
    expected_keys = {"precision", "recall", "f1", "accuracy"}
    assert all(k in class_metrics for k in expected_keys), "Ключи classification_metrics отсутствуют"

    precision = class_metrics["precision"].item()
    recall = class_metrics["recall"].item()
    f1 = class_metrics["f1"].item()

    assert 0 <= precision <= 1, "Precision вне диапазона [0, 1]"
    assert 0 <= recall <= 1, "Recall вне диапазона [0, 1]"
    assert 0 <= f1 <= 1, "f1 вне диапазона [0, 1]"

