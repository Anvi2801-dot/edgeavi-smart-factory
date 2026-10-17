from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DefectDataset(Dataset):
    def __init__(self, split_dir: str | Path, height: int, width: int, train: bool = False):
        self.dir = Path(split_dir)
        self.names = sorted(p.name for p in (self.dir / "images").glob("*.png"))
        self.h, self.w, self.train = height, width, train

    def __len__(self):
        return len(self.names)

    def load(self, i: int):
        img = cv2.imread(str(self.dir / "images" / self.names[i]))
        mask = cv2.imread(str(self.dir / "masks" / self.names[i]), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load(i)
        if self.train and np.random.rand() < 0.5:        # horizontal flip
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self.train and np.random.rand() < 0.5:        # vertical flip
            img, mask = img[::-1], mask[::-1]
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)
        mask = (mask >= 100).astype(np.float32)           # keep thin scratches alive
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(mask)[None]
