from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


class LeavingRichProgressBar(RichProgressBar):
    def on_train_end(self, trainer, pl_module):
        if self.progress:
            self.progress.stop()
        super().on_train_end(trainer, pl_module)
