import lightning as L
from .constant import SEED
from .dataset import *
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from enum import Enum


class TrainMode(Enum):
    CONTRASTIVE = "contrastive"
    ARCFACE = "arcface"


class FaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        rec_path,
        idx_path,
        train_ids,
        test_ids,
        eval_bins,  # dict: {'lfw': 'eval/lfw.bin', ...}
        train_transform=None,
        val_transform=None,
        batch_size=64,
        num_workers=4,
        seed=SEED,
    ):
        super().__init__()
        self.rec_path = rec_path
        self.idx_path = idx_path
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.eval_bins = eval_bins
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        L.seed_everything(self.seed)

        if stage in ("fit", None):
            self.train_ds = RecDataset(
                self.rec_path,
                self.idx_path,
                allowed_ids=self.train_ids,
                transform=self.train_transform,
            )

        if stage in ("test", None):
            # 1. rec test split
            self.rec_test_ds = RecDataset(
                self.rec_path,
                self.idx_path,
                allowed_ids=self.test_ids,
                transform=self.val_transform,
            )
            # 2–6. eval .bin benchmarks
            self.bin_test_ds = {
                name: BinDataset(path, self.val_transform)
                for name, path in self.eval_bins.items()
            }

    def _make_loader(self, ds, shuffle=False):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=g,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_ds, shuffle=True)

    def test_dataloader(self):
        # returns a list — Lightning runs test_step once per loader
        # order: rec_test, lfw, cfp_fp, agedb_30, sllfw, talfw
        loaders = [self._make_loader(self.rec_test_ds)]
        for name in self.eval_bins:
            loaders.append(self._make_loader(self.bin_test_ds[name]))
        return loaders


class KFoldFaceDataModule(FaceDataModule):
    def __init__(
        self,
        *args,
        n_folds=5,
        fold=0,
        mode: TrainMode = TrainMode.CONTRASTIVE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert 0 <= fold < n_folds, f"fold must be 0–{n_folds-1}"
        self.n_folds = n_folds
        self.fold = fold
        self.mode = mode

    def setup(self, stage=None):
        L.seed_everything(self.seed)

        if stage in ("fit", None):
            # Build full train dataset first (no fold split yet)
            full_ds = RecDataset(
                self.rec_path,
                self.idx_path,
                allowed_ids=self.train_ids,
                transform=None,  # transform applied per-subset below
            )
            unique_ids = sorted(self.train_ids)
            id_to_idx = {pid: i for i, pid in enumerate(unique_ids)}

            # Labels for stratification — preserve class balance across folds
            all_labels = [full_ds.samples[i][1] for i in range(len(full_ds))]

            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )
            splits = list(skf.split(range(len(full_ds)), all_labels))

            train_idx, val_idx = splits[self.fold]

            train_subset = TransformSubset(
                full_ds, train_idx, self.train_transform, label_remap=id_to_idx
            )
            val_subset = TransformSubset(full_ds, val_idx, self.val_transform)

            if self.mode == TrainMode.CONTRASTIVE:
                self.train_ds = SiamesePairDataset(train_subset, seed=self.seed)
            elif self.mode == TrainMode.ARCFACE:
                self.train_ds = train_subset

            self.val_ds = SiamesePairDataset(val_subset, seed=self.seed)

        if stage in ("test", None):
            self.rec_test_ds = SiamesePairDataset(
                RecDataset(
                    self.rec_path,
                    self.idx_path,
                    allowed_ids=self.test_ids,
                    transform=self.val_transform,
                ),
                seed=self.seed,
            )
            self.bin_test_ds = {
                name: BinDataset(path, self.val_transform)
                for name, path in self.eval_bins.items()
            }

    def train_dataloader(self):
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_ds, shuffle=False)
