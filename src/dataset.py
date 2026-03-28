import struct, io, pickle, torch, random
from PIL import Image
from torch.utils.data import Dataset
from .constant import _IR_BUFFER, _IR_FORMAT, _IR_SIZE, SEED
from collections import defaultdict

VALID_MIN_IMG_SIZE = 100


class RecDataset(Dataset):
    def __init__(self, rec_path, idx_path, allowed_ids=None, transform=None):
        """
        allowed_ids : set of int — if None, loads everything
        """
        self.rec_path = rec_path
        self.transform = transform
        self._rec_file = None

        raw_idx = {}
        with open(idx_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, offset = line.split("\t")
                raw_idx[int(key)] = int(offset)

        # Filter: read every header, keep only allowed identities
        self.samples = []  # list of (offset, label)
        with open(rec_path, "rb") as f:
            for offset in raw_idx.values():
                f.seek(offset)
                _, length = struct.unpack("II", f.read(_IR_BUFFER))  # magic + length
                _, label, _, _ = struct.unpack(_IR_FORMAT, f.read(_IR_SIZE))

                img_size = length - _IR_SIZE
                if img_size < VALID_MIN_IMG_SIZE:
                    continue

                pid = int(label)
                if allowed_ids is None or pid in allowed_ids:
                    self.samples.append((offset, pid))

    def _open(self):
        if self._rec_file is None:
            self._rec_file = open(self.rec_path, "rb")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        self._open()
        offset, label = self.samples[i]
        self._rec_file.seek(offset)

        magic, length = struct.unpack("II", self._rec_file.read(_IR_BUFFER))
        flag, lbl, _, _ = struct.unpack(_IR_FORMAT, self._rec_file.read(_IR_SIZE))

        img_size = (
            length - _IR_SIZE
        )  # derive size from length field, not a separate read
        if img_size <= 0:
            raise ValueError(
                f"Invalid img_size={img_size} at offset={offset}, length={length}"
            )
        img_bytes = self._rec_file.read(img_size)

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Failed to decode image at offset={offset}, img_size={img_size}, length={length}, _IR_SIZE={_IR_SIZE}"
            ) from e

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class BinDataset(Dataset):
    def __init__(self, bin_path, transform=None):
        self.transform = transform
        with open(bin_path, "rb") as f:
            self.bins, self.issame = pickle.load(f, encoding="bytes")

    def __len__(self):
        return len(self.issame)

    def _decode(self, raw):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img) if self.transform else img

    def __getitem__(self, i):
        img_a = self._decode(self.bins[i * 2])
        img_b = self._decode(self.bins[i * 2 + 1])
        same = torch.tensor(int(self.issame[i]), dtype=torch.float32)
        return img_a, img_b, same


class TransformSubset(Dataset):
    """
    Wraps a Subset so train and val can have
    different transforms on the same underlying dataset.
    """

    def __init__(self, dataset, indices, transform=None, label_remap=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.label_remap = label_remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        # img here is a raw PIL Image (transform=None in full_ds)
        if self.transform:
            img = self.transform(img)
        if self.label_remap is not None:
            label = torch.tensor(self.label_remap[label.item()], dtype=torch.long)
        return img, label


class SiamesePairDataset(Dataset):
    """
    Wraps any RecDataset (or TransformSubset of one).
    Returns (img_a, img_b, cosine_label) where
    cosine_label = +1 (same) or -1 (different)
    as required by CosineEmbeddingLoss.
    """

    def __init__(self, dataset, n_pairs=None, seed=SEED):
        self.dataset = dataset
        self.seed = seed

        self.label_to_idx = defaultdict(list)

        # TransformSubset → access underlying RecDataset.samples directly
        if isinstance(dataset, TransformSubset):
            src = dataset.dataset  # the RecDataset
            indices = dataset.indices
            for i, raw_idx in enumerate(indices):
                _, label = src.samples[raw_idx]
                self.label_to_idx[label].append(i)
        else:
            # plain RecDataset
            for i, (_, label) in enumerate(dataset.samples):
                self.label_to_idx[label].append(i)

        self.labels = list(self.label_to_idx.keys())
        self.n_pairs = n_pairs or len(dataset)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, i):
        rng = random.Random(self.seed + i + self.epoch * self.n_pairs)
        same = rng.random() > 0.5
        label_a = rng.choice(self.labels)
        idx_a = rng.choice(self.label_to_idx[label_a])
        img_a, _ = self.dataset[idx_a]

        if same and len(self.label_to_idx[label_a]) > 1:
            idx_b = rng.choice(self.label_to_idx[label_a])
            label = torch.tensor(1.0)
        else:
            label_b = rng.choice([l for l in self.labels if l != label_a])
            idx_b = rng.choice(self.label_to_idx[label_b])
            label = torch.tensor(0.0)

        img_b, _ = self.dataset[idx_b]
        return img_a, img_b, label
