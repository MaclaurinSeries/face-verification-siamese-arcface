import os, torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning as L
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from src.loss import *


class SiameseLightningModule(L.LightningModule):
    def __init__(self, backbone: nn.Module, lr=1e-4, unif_lambda=1.0, test_names=[]):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.test_names = test_names
        self.unif_lambda = unif_lambda

        # Metrics (binary: same=1, diff=0)
        metric_args = dict(task="binary")
        for split in ("val", "test"):
            setattr(self, f"{split}_acc", Accuracy(**metric_args))
            setattr(self, f"{split}_f1", F1Score(**metric_args, average="macro"))
            setattr(self, f"{split}_auc", AUROC(**metric_args))
            setattr(self, f"{split}_ap", AveragePrecision(**metric_args))

        # Buffers to accumulate test outputs per dataloader
        self._test_preds = {}
        self._test_labels = {}

    def _contrastive_loss(self, emb_a, emb_b, label, margin=1.0):
        dist = F.pairwise_distance(emb_a, emb_b)
        same_loss = label * dist.pow(2)
        diff_loss = (1 - label) * F.relu(margin - dist).pow(2)
        return (same_loss + diff_loss).mean()

    def _alignment_loss(self, emb_a, emb_b, alpha=2):
        return (emb_a - emb_b).norm(p=2, dim=1).pow(alpha).mean()

    def _uniformity_loss(self, emb, t=2):
        return torch.pdist(emb, p=2).pow(2).mul(-t).exp().mean().log()

    def _embed(self, img):
        return self.backbone(img)

    def _shared_pair_step(self, batch, batch_idx):
        img_a, img_b, label = batch
        emb_a = self._embed(img_a).float()
        emb_b = self._embed(img_b).float()

        loss = self._contrastive_loss(emb_a, emb_b, label)

        dist = F.pairwise_distance(emb_a, emb_b)
        prob = 1 / (1 + dist)

        return loss, prob, label.long()

    def on_train_epoch_start(self):
        ds = self.trainer.train_dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_pair_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prob, label = self._shared_pair_step(batch, batch_idx)
        self.val_acc.update(prob, label)
        self.val_f1.update(prob, label)
        self.val_auc.update(prob, label)
        self.val_ap.update(prob, label)
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val/acc": self.val_acc.compute(),
                "val/f1": self.val_f1.compute(),
                "val/auc": self.val_auc.compute(),
                "val/ap": self.val_ap.compute(),
            },
            prog_bar=True,
        )
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()
        self.val_ap.reset()

    def on_test_start(self):
        self._test_preds = {n: [] for n in self.test_names}
        self._test_labels = {n: [] for n in self.test_names}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _, prob, label = self._shared_pair_step(batch, batch_idx)
        name = self.test_names[dataloader_idx]
        self._test_preds[name].append(prob.cpu())
        self._test_labels[name].append(label.cpu())
        self.test_acc.update(prob, label)
        self.test_f1.update(prob, label)
        self.test_auc.update(prob, label)
        self.test_ap.update(prob, label)

    def on_test_epoch_end(self):
        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)

        for name in self.test_names:
            probs = torch.cat(self._test_preds[name]).numpy()
            labels = torch.cat(self._test_labels[name]).numpy()
            preds = (probs >= 0.5).astype(int)

            # ── CSV ──────────────────────────────────────────
            pd.DataFrame(
                {
                    "prob_same": probs,
                    "pred": preds,
                    "label": labels,
                }
            ).to_csv(f"{log_dir}/{name}_predictions.csv", index=False)

            # ── Confusion matrix ─────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                labels, preds, display_labels=["diff", "same"], ax=ax
            )
            ax.set_title(f"{name} — confusion matrix")
            fig.savefig(f"{log_dir}/{name}_confusion_matrix.png", bbox_inches="tight")
            plt.close(fig)

            # ── ROC curve ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=name)
            ax.set_title(f"{name} — ROC curve")
            fig.savefig(f"{log_dir}/{name}_roc_curve.png", bbox_inches="tight")
            plt.close(fig)

            # ── PR curve ─────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            PrecisionRecallDisplay.from_predictions(labels, probs, ax=ax, name=name)
            ax.set_title(f"{name} — PR curve")
            fig.savefig(f"{log_dir}/{name}_pr_curve.png", bbox_inches="tight")
            plt.close(fig)

        # Log final test metrics
        self.log_dict(
            {
                "test/acc": self.test_acc.compute(),
                "test/f1": self.test_f1.compute(),
                "test/auc": self.test_auc.compute(),
                "test/ap": self.test_ap.compute(),
            }
        )
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auc.reset()
        self.test_ap.reset()

    def configure_optimizers(self):
        if hasattr(self.backbone, "features") and hasattr(self.backbone, "embedder"):
            # EfficientNet — differential LR
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.backbone.features.parameters(),
                        "lr": self.hparams.lr * 0.1,
                    },
                    {
                        "params": self.backbone.embedder.parameters(),
                        "lr": self.hparams.lr,
                    },
                ]
            )
        else:
            # SimpleCNN
            optimizer = torch.optim.AdamW(
                self.backbone.parameters(), lr=self.hparams.lr
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


class ArcFaceLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone,
        num_classes,
        embed_dim=256,
        lr=3e-4,
        s=64.0,
        m=0.5,
        test_names=[],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.loss_fn = ArcFaceLoss(embed_dim, num_classes, s=s, m=m)
        self.test_names = test_names

        metric_args = dict(task="binary")
        for split in ("val", "test"):
            setattr(self, f"{split}_acc", Accuracy(**metric_args))
            setattr(self, f"{split}_f1", F1Score(**metric_args, average="macro"))
            setattr(self, f"{split}_auc", AUROC(**metric_args))
            setattr(self, f"{split}_ap", AveragePrecision(**metric_args))

        self._test_preds = {}
        self._test_labels = {}

    def _embed(self, img):
        return F.normalize(self.backbone(img), p=2, dim=1)

    def _eval_pair_step(self, batch):
        img_a, img_b, label = batch
        emb_a = self._embed(img_a).float()
        emb_b = self._embed(img_b).float()
        dist = F.pairwise_distance(emb_a, emb_b)
        prob = 1 / (1 + dist)
        return prob, label.long()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch  # single images + identity label
        emb = self._embed(imgs).float()
        loss = self.loss_fn(emb, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prob, label = self._eval_pair_step(batch)
        self.val_acc.update(prob, label)
        self.val_f1.update(prob, label)
        self.val_auc.update(prob, label)
        self.val_ap.update(prob, label)

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val/acc": self.val_acc.compute(),
                "val/f1": self.val_f1.compute(),
                "val/auc": self.val_auc.compute(),
                "val/ap": self.val_ap.compute(),
            },
            prog_bar=True,
        )
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()
        self.val_ap.reset()

    def on_test_start(self):
        self._test_preds = {n: [] for n in self.test_names}
        self._test_labels = {n: [] for n in self.test_names}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        prob, label = self._eval_pair_step(batch)
        name = self.test_names[dataloader_idx]
        self._test_preds[name].append(prob.cpu())
        self._test_labels[name].append(label.cpu())
        self.test_acc.update(prob, label)
        self.test_f1.update(prob, label)
        self.test_auc.update(prob, label)
        self.test_ap.update(prob, label)

    def on_test_epoch_end(self):
        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)

        for name in self.test_names:
            probs = torch.cat(self._test_preds[name]).numpy()
            labels = torch.cat(self._test_labels[name]).numpy()
            preds = (probs >= 0.5).astype(int)

            # ── CSV ──────────────────────────────────────────
            pd.DataFrame(
                {
                    "prob_same": probs,
                    "pred": preds,
                    "label": labels,
                }
            ).to_csv(f"{log_dir}/{name}_predictions.csv", index=False)

            # ── Confusion matrix ─────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                labels, preds, display_labels=["diff", "same"], ax=ax
            )
            ax.set_title(f"{name} — confusion matrix")
            fig.savefig(f"{log_dir}/{name}_confusion_matrix.png", bbox_inches="tight")
            plt.close(fig)

            # ── ROC curve ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=name)
            ax.set_title(f"{name} — ROC curve")
            fig.savefig(f"{log_dir}/{name}_roc_curve.png", bbox_inches="tight")
            plt.close(fig)

            # ── PR curve ─────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 4))
            PrecisionRecallDisplay.from_predictions(labels, probs, ax=ax, name=name)
            ax.set_title(f"{name} — PR curve")
            fig.savefig(f"{log_dir}/{name}_pr_curve.png", bbox_inches="tight")
            plt.close(fig)

        # Log final test metrics
        self.log_dict(
            {
                "test/acc": self.test_acc.compute(),
                "test/f1": self.test_f1.compute(),
                "test/auc": self.test_auc.compute(),
                "test/ap": self.test_ap.compute(),
            }
        )
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auc.reset()
        self.test_ap.reset()

    def configure_optimizers(self):
        if hasattr(self.backbone, "features") and hasattr(self.backbone, "embedder"):
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.backbone.features.parameters(),
                        "lr": self.hparams.lr * 0.5,
                    },
                    {
                        "params": self.backbone.embedder.parameters(),
                        "lr": self.hparams.lr,
                    },
                    {"params": self.loss_fn.parameters(), "lr": self.hparams.lr},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.backbone.parameters(), "lr": self.hparams.lr},
                    {"params": self.loss_fn.parameters(), "lr": self.hparams.lr},
                ]
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[5]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/auc",
        }
