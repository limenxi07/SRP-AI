import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torchvision.io import read_image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import zipfile
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        annotations: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (128, 128),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_path # root directory of the dataset
        self.annotations = annotations
        self.train_batch_size = train_batch_size # batch size to use during training
        self.val_batch_size = val_batch_size # batch size to use during validation
        self.patch_size = patch_size # size of crop to take from original images
        self.num_workers = num_workers # number of parallel workers to load data items
        self.pin_memory = pin_memory # whether prepared items should be loaded into pinned memory

    def setup(self, stage: Optional[str] = None) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage()]
        ),
        
        self.train_dataset = MyDataset(
            data_dir=self.data_dir,
            annotations_file=self.annotations,
            transform=transform,
        )
        
        self.val_dataset = MyDataset(
            data_dir=self.data_dir,
            annotations_file=self.annotations,
            transform=transform,
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     