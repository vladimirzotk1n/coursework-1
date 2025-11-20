from ..config import PATIENCE


class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.count_no_improve = 0
        self.stop_training = False

    def check(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.count_no_improve = 0
        else:
            self.count_no_improve += 1
            if self.count_no_improve >= self.patience:
                self.stop_training = True

