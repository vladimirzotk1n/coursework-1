import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from coursework_1.modeling.early_stopping import EarlyStopping
from coursework_1.config import NUM_EPOCHS, DEVICE, CHECKPOINT_PATH
from .metrics import Metrics


def train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    early_stopper = EarlyStopping()
    metrics = Metrics()
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "train_dice": [], "val_dice": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": []
    }

    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        loss_train, iou_train, dice_train = 0.0, 0.0, 0.0
        acc_train, prec_train, rec_train, f1_train = 0.0, 0.0, 0.0, 0.0

        for imgs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, labels.float())
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            train_preds = preds.detach()
            train_labels = labels.detach()

            iou_vals = metrics.iou(train_preds, train_labels)
            dice_vals = metrics.dice(train_preds, train_labels)
            iou_train += iou_vals.mean().item()
            dice_train += dice_vals.mean().item()

            cls_metrics = metrics.classification_metrics(train_preds, train_labels)
            acc_train += cls_metrics['accuracy'].item()
            prec_train += cls_metrics['precision'].item()
            rec_train += cls_metrics['recall'].item()
            f1_train += cls_metrics['f1'].item()

        n_train_batches = len(train_dataloader)
        history["train_loss"].append(loss_train / n_train_batches)
        history["train_iou"].append(iou_train / n_train_batches)
        history["train_dice"].append(dice_train / n_train_batches)
        history["train_accuracy"].append(acc_train / n_train_batches)
        history["train_precision"].append(prec_train / n_train_batches)
        history["train_recall"].append(rec_train / n_train_batches)
        history["train_f1"].append(f1_train / n_train_batches)

        print(f"Train - loss: {history['train_loss'][-1]:.4f}, IoU: {history['train_iou'][-1]:.4f}, "
              f"Dice: {history['train_dice'][-1]:.4f}, Acc: {history['train_accuracy'][-1]:.4f}, "
              f"Prec: {history['train_precision'][-1]:.4f}, Rec: {history['train_recall'][-1]:.4f}, "
              f"F1: {history['train_f1'][-1]:.4f}")

        model.eval()
        loss_val, iou_val, dice_val = 0.0, 0.0, 0.0
        acc_val, prec_val, rec_val, f1_val = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)

                loss = criterion(preds, labels.float())
                loss_val += loss.item()

                val_preds = preds.detach()
                val_labels = labels.detach()

                iou_vals = metrics.iou(val_preds, val_labels)
                dice_vals = metrics.dice(val_preds, val_labels)
                iou_val += iou_vals.mean().item()
                dice_val += dice_vals.mean().item()

                cls_metrics = metrics.classification_metrics(val_preds, val_labels)
                acc_val += cls_metrics['accuracy'].item()
                prec_val += cls_metrics['precision'].item()
                rec_val += cls_metrics['recall'].item()
                f1_val += cls_metrics['f1'].item()

        n_val_batches = len(val_dataloader)
        history["val_loss"].append(loss_val / n_val_batches)
        history["val_iou"].append(iou_val / n_val_batches)
        history["val_dice"].append(dice_val / n_val_batches)
        history["val_accuracy"].append(acc_val / n_val_batches)
        history["val_precision"].append(prec_val / n_val_batches)
        history["val_recall"].append(rec_val / n_val_batches)
        history["val_f1"].append(f1_val / n_val_batches)

        print(f"Val   - loss: {history['val_loss'][-1]:.4f}, IoU: {history['val_iou'][-1]:.4f}, "
              f"Dice: {history['val_dice'][-1]:.4f}, Acc: {history['val_accuracy'][-1]:.4f}, "
              f"Prec: {history['val_precision'][-1]:.4f}, Rec: {history['val_recall'][-1]:.4f}, "
              f"F1: {history['val_f1'][-1]:.4f}")

        scheduler.step(history["val_loss"][-1])

        early_stopper.check(history["val_loss"][-1])
        if early_stopper.stop_training:
            print("Early stopping triggered.")
            break

        if history["val_loss"][-1] <= early_stopper.best_loss:
            print(f"Model saved with val_loss = {history['val_loss'][-1]:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history["val_loss"][-1]
            }, CHECKPOINT_PATH)

    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    ax_loss.plot(history["train_loss"], label="train_loss")
    ax_loss.plot(history["val_loss"], label="val_loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()

    fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
    ax_metrics.plot(history["train_iou"], label="train_iou")
    ax_metrics.plot(history["val_iou"], label="val_iou")
    ax_metrics.plot(history["train_dice"], label="train_dice")
    ax_metrics.plot(history["val_dice"], label="val_dice")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Metric")
    ax_metrics.set_title("IoU and Dice Metrics")
    ax_metrics.legend()


    plt.show()

    return history

