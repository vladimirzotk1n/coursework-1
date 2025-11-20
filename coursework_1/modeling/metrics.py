import torch


class Metrics:
    def __init__(self, smooth=1.0, eps=1e-8):
        self.smooth = smooth
        self.eps = eps

    def dice(self, preds, labels, per_class=True):
        preds = torch.softmax(preds, dim=1)
        dot_product = (preds * labels).sum(dim=(0, 2, 3))
        preds_sum = preds.sum(dim=(0, 2, 3))
        labels_sum = labels.sum(dim=(0, 2, 3))
        dice_per_class = 2.0 * dot_product / (preds_sum + labels_sum + self.smooth)

        return dice_per_class if per_class else dice_per_class.mean()

    def iou(self, preds, labels, per_class=True):
        preds = torch.softmax(preds, dim=1)
        dot_product = (preds * labels).sum(dim=(0, 2, 3))
        preds_sum = preds.sum(dim=(0, 2, 3))
        labels_sum = labels.sum(dim=(0, 2, 3))
        iou_per_class = dot_product / (preds_sum + labels_sum - dot_product + self.smooth)

        return iou_per_class if per_class else iou_per_class.mean()

    def classification_metrics(self, preds, labels):
        preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        labels = torch.argmax(labels, dim=1)
        num_classes = torch.max(labels).item() + 1

        tp, fp, fn, tn = [], [], [], []
        for c in range(num_classes):
            pred_c = (preds == c)
            label_c = (labels == c)
            tp.append((pred_c & label_c).sum())
            fp.append((pred_c & ~label_c).sum())
            fn.append((~pred_c & label_c).sum())
            tn.append((~pred_c & ~label_c).sum())

        tp = torch.tensor(tp, dtype=torch.float32)
        fp = torch.tensor(fp, dtype=torch.float32)
        fn = torch.tensor(fn, dtype=torch.float32)
        tn = torch.tensor(tn, dtype=torch.float32)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.eps)

        return {
            "accuracy": accuracy.mean(),
            "precision": precision.mean(),
            "recall": recall.mean(),
            "f1": f1.mean()
        }

