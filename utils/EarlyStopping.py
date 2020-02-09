class EarlyStopping:
    """Early stopping tracks the validation loss and
    stops the training if the validation loss doesn't improve over consecutive epochs."""

    def __init__(self, patience=60):
        """
        Args:
            patience (int): Patience (in terms of number of epochs) to check validation loss improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_val = None
        self.best_epoch = None
        self.stop = False

    def check_improvement(self, val_loss, epoch):
        if self.best_val is None:
            self.best_val = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_val:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_val = val_loss
            self.best_epoch = epoch
            self.counter = 0
        return self.stop
