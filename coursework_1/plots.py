import torch
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from .config import DEVICE
from .modeling.metrics import Metrics


def plot_metrics(model, dataloader):
    metrics = Metrics()
    dice, iou, precision, recall, f1 = [], [], [], [], []
    model.eval()

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(imgs)

            dice.append(metrics.dice(preds, masks, per_class=False).cpu())
            iou.append(metrics.iou(preds, masks, per_class=False).cpu())
            classification_metrics = metrics.classification_metrics(preds, masks)
            precision.append(classification_metrics["precision"].cpu())
            recall.append(classification_metrics["recall"].cpu())
            f1.append(classification_metrics["f1"].cpu())

    metrics_dict = {
        'Dice': dice,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    for metric_name, values in metrics_dict.items():
        plt.figure(figsize=(10, 6))

        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)

        n, bins, patches = plt.hist(values, bins=30, alpha=0.7, color='lightblue')

        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Std: {std_val:.4f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--')

        plt.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='yellow')

        plt.title(f'{metric_name} распределение\Среднее: {mean_val:.4f} ± {std_val:.4f}')
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def visualize_prediction(model, dataset):
    i = random.randint(0, len(dataset) - 1)
    img, mask = dataset[i]

    img_batch, mask_batch = img.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(img_batch)
        pred = torch.argmax(pred, dim=1)
        mask_batch = torch.argmax(mask_batch, dim=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(torch.mean(img, dim=0).squeeze().cpu(), cmap='gray')
    ax[0].set_title("МРТ снимок")
    ax[0].set_axis_off()

    ax[1].imshow(pred.squeeze(0).cpu())
    ax[1].set_title("Предсказанная маска")
    ax[1].set_axis_off()

    ax[2].imshow(mask_batch.squeeze(0).cpu())
    ax[2].set_title("Ground Truth маска")
    ax[2].set_axis_off()

    plt.show()