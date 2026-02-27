import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pytorch_lightning as pl
from typing import List, Tuple, Optional

from monai.transforms import (
    Compose,
    Resized,
    Rand3DElasticd,
    RandFlipd,
    RandAffined,
    NormalizeIntensityd,
    ToTensord,
    RandBiasFieldd,
    RandRicianNoised,
    ScaleIntensityd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
)

def get_shared_spatial_transform(spatial_size: Tuple[int, int, int]):
    """
    Returns the list of spatial transforms for MRI volumes.
    """
    return Compose([
        LoadImaged(keys=["volume"], image_only=True),
        EnsureChannelFirstd(keys=["volume"]),
        Resized(keys=["volume"], spatial_size=spatial_size, mode="trilinear"),
        Rand3DElasticd(keys=["volume"], sigma_range=(5, 8), magnitude_range=(100, 200), prob=0.3),
        RandFlipd(keys=["volume"], spatial_axis=[2], prob=0.5),
        RandAffined(
            keys=["volume"],
            rotate_range=((-0.3, 0.3),) * 3,
            translate_range=((-10, 10),) * 3,
            scale_range=((0.85, 1.15),) * 3,
            padding_mode="border",
            prob=0.7,
        ),
        NormalizeIntensityd(keys=["volume"], nonzero=True, channel_wise=True),
        ToTensord(keys=["volume"]),
    ])

def get_validation_transform(spatial_size: Tuple[int, int, int]):
    return Compose([
        LoadImaged(keys=["volume"], image_only=True),
        EnsureChannelFirstd(keys=["volume"]),
        Resized(keys=["volume"], spatial_size=spatial_size, mode="trilinear"),
        NormalizeIntensityd(keys=["volume"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["volume"], minv=0.0, maxv=1.0),
        ToTensord(keys=["volume"]),
    ])

def get_independent_intensity_transform():
    return Compose([
        RandBiasFieldd(keys=["volume"], coeff_range=(0.3, 0.6), prob=0.3),
        RandRicianNoised(keys=["volume"], prob=0.3, mean=0.0, std=0.03, channel_wise=True),
        RandGaussianSmoothd(keys=["volume"], prob=0.5, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandAdjustContrastd(keys=["volume"], prob=0.5, gamma=(0.5, 2.0)),
        RandShiftIntensityd(keys=["volume"], offsets=(-0.2, 0.2), prob=0.5),
        #ScaleIntensityd(keys=["volume"], minv=0.0, maxv=1.0),
        ToTensord(keys=["volume"]),
    ])


class MRITextDataset(Dataset):
    """ 
    Main dataset class to load images and text blurbs 
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        text_dir: str,
        spatial_transform: Optional[Compose] = None,
        intensity_transform: Optional[Compose] = None,
    ):
        self.df = pd.read_csv(csv_path, dtype={"pat_id": str, "scandates": str})
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.spatial_transform = spatial_transform
        self.intensity_transform = intensity_transform

    def __len__(self):
        return len(self.df)

    def _get_volume_path(self, pat_id: str, date: str) -> str:
        fname = f"{pat_id}_{date}.nii.gz"
        return os.path.join(self.image_dir, fname)

    def _load_text(self, pat_id: str, dates: List[str]) -> str:
        txt_name = f"{pat_id}_{'_'.join(dates)}.txt"
        path = os.path.join(self.text_dir, txt_name)
        try:
            with open(path, "r") as f:
                return f.read().strip() # return text string 
        except Exception:
            return ""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pat_id = str(row["pat_id"])
        dates = row["scandates"].split("-")
        label = int(row["label"])

        volume_paths = [self._get_volume_path(pat_id, d) for d in dates]
        items = [{"volume": path} for path in volume_paths]

        if self.spatial_transform:
            seed = random.randint(0, 2**31 - 1)
            self.spatial_transform.set_random_state(seed)  # fix the seed to ensure all scans get same spatial transforms
            items = [self.spatial_transform(it) for it in items]

        if self.intensity_transform:
            items = [self.intensity_transform(it) for it in items] # intensity transforms can vary between scans  

        vols_tensors = [it["volume"] for it in items]
        for i, vol in enumerate(vols_tensors):
            if vol.shape[0] != 1:
                print(f"WARNING: Volume {i} for patient {pat_id}, date {dates[i]} has {vol.shape[0]} channels instead of 1")

        text_blob = self._load_text(pat_id, dates)
        return vols_tensors, text_blob, label

# custom collate function to pad zero vols for uneven scandates 
def collate_fn(batch):
    vols_list, texts, labels = zip(*batch)
    B = len(vols_list)
    C, D, H, W = vols_list[0][0].shape
    T_max = max(len(v) for v in vols_list)

    vols_tensor = torch.zeros(B, T_max, C, D, H, W, dtype=torch.float32)
    pad_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, vs in enumerate(vols_list):   ## pad zero volumes where the scans are not availaible 
        L = len(vs)
        vols_tensor[i, :L] = torch.stack(vs)
        pad_mask[i, :L] = True

    labels_t = torch.tensor(labels, dtype=torch.long)
    return vols_tensor, list(texts), pad_mask, labels_t

class MRITextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        image_dir: str,
        text_dir: str,
        cfg: dict,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        ss = self.cfg.image_size
        spatial_size = tuple(ss) if isinstance(ss, (list, tuple)) else (ss, ss, ss)
        train_spatial = get_shared_spatial_transform(spatial_size)
        val_spatial = get_validation_transform(spatial_size)
        intensity = get_independent_intensity_transform()

        self.train_ds = MRITextDataset(
            csv_path=self.train_csv,
            image_dir=self.image_dir,
            text_dir=self.text_dir,
            spatial_transform=train_spatial,
            intensity_transform=intensity,
        )
        self.val_ds = MRITextDataset(
            csv_path=self.val_csv,
            image_dir=self.image_dir,
            text_dir=self.text_dir,
            spatial_transform=val_spatial,
            intensity_transform=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

