import torch
from tqdm import tqdm
from models import UNet
from ..config import DEVICE, CHECKPOINT_PATH
from .metrics import Metrics


def evaluate_model_on_test(model, test_dataloader):
    metrics = Metrics()

    predictions = []
    labels = []

    model.to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    with torch.no_grad():
        for imgs, masks in tqdm(test_dataloader):
            imgs = imgs.to(DEVICE)
            preds = model(imgs).detach().cpu()
            predictions.append(preds)
            labels.append(masks)

    predictions_tensor = torch.cat(predictions, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    dice_per_class = metrics.dice(predictions_tensor, labels_tensor, per_class=True).tolist()
    dice_mean = metrics.dice(predictions_tensor, labels_tensor, per_class=False).item()

    iou_per_class = metrics.iou(predictions_tensor, labels_tensor, per_class=True).tolist()
    iou_mean = metrics.iou(predictions_tensor, labels_tensor, per_class=False).item()

    classification_metrics = metrics.classification_metrics(predictions_tensor, labels_tensor)

    precision = classification_metrics["precision"].item()
    recall = classification_metrics["recall"].item()
    f1 = classification_metrics["f1"].item()

    print("="*20, "TEST REPORT", "="*20)
    print(f"Dice per class: {dice_per_class}")
    print(f"Dice mean: {dice_mean}")

    print(f"IoU per class: {iou_per_class}")
    print(f"IoU mean: {iou_mean}")

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("="*51)


