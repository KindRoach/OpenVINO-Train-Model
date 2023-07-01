import os
from typing import List

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image = self.images[idx]
        label = self.labels[idx]

        # Apply the transformation to the image
        image = self.transform(image)
        return image, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, batch_size: int, image_shape: (int, int)):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dataset = datasets.load_dataset(self.dataset_name, cache_dir="output/dataset")

        mean, std = calculate_dataset_statistics(self.dataset["train"]["img"], self.image_shape)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def setup(self, stage: str):
        dataset_train = self.dataset["train"]
        train_data = ImageDataset(dataset_train["img"], dataset_train["label"], self.transform)

        train_set_size = int(len(train_data) * 0.8)
        valid_set_size = len(train_data) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.valid_set = random_split(train_data, [train_set_size, valid_set_size], generator=seed)

        dataset_test = self.dataset["test"]
        self.test_set = ImageDataset(dataset_test["img"], dataset_test["label"], self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_set, self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=os.cpu_count())


def calculate_dataset_statistics(imgs: List[ImageFile], image_shape: (int, int)) -> (np.ndarray, np.ndarray):
    """
    Reference: https://kozodoi.me/blog/20210308/compute-image-stats
    :param imgs:
    :param img_size:
    :return: mean and std of dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor()
    ])

    p_sum = torch.zeros(3, dtype=torch.float64)
    p_sum_sq = torch.zeros(3, dtype=torch.float64)
    for img in tqdm(imgs, desc="Calculating mean and std"):
        img = transform(img)
        p_sum += img.sum(dim=(1, 2))
        p_sum_sq += (img ** 2).sum(dim=(1, 2))

    count = len(imgs) * image_shape[0] * image_shape[1]
    total_mean = p_sum / count
    total_var = (p_sum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean.numpy(), total_std.numpy()
