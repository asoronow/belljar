import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import os
import pickle
import cv2
from PIL import Image
from math import exp
import numpy as np
from slice_atlas import slice_3d_volume


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return self.relu(x)


class TissueAutoencoder(nn.Module):
    def __init__(self):
        super(TissueAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32 * 32 * 128),
            nn.ReLU(),
        )

        # Decoder
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1)

    def encode_to_latent(self, x):
        """
        Passes the input through the encoder and bottleneck to produce the latent vector.

        Args:
        - x (torch.Tensor): The input tensor

        Returns:
        - torch.Tensor: The latent representation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x, idx1 = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x, idx2 = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x, idx3 = self.pool3(x)

        # Pass through the bottleneck and get the latent representation
        x = self.bottleneck[0:2](
            x
        )  # We only need up to the first linear layer to get the latent vector

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x, idx1 = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x, idx2 = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x, idx3 = self.pool3(x)

        # Pass through the bottleneck
        x = self.bottleneck(x)

        x = x.view(-1, 128, 32, 32)  # Reshape to match decoder's expected input shape

        x = self.unpool1(x, idx3)
        x = self.deconv1(x)
        x = self.bn4(x)

        x = self.unpool2(x, idx2)
        x = self.deconv2(x)
        x = self.bn5(x)

        x = self.unpool3(x, idx1)
        x = self.deconv3(x)
        x = self.bn6(x)

        return x


class TissuePredictor(nn.Module):
    def __init__(self):
        super(TissuePredictor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 3)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No activation here as it's a regression output

        return x


class AtlasDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.endswith(".png")
        ]

        # Default transform: Convert images to grayscale and then to tensors
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [transforms.Grayscale(), transforms.ToTensor()]
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)
        if image.size != (256, 256):
            image = image.resize((256, 256))

        if self.transform:
            image = self.transform(image)

        return image


class AngledAtlasDataset(Dataset):
    """
    Dataset of n randomly generate images from the provided atlas at random angles and positions
    """

    def __init__(self, data_path, transform=None):
        self.transform = transform

        self.data_path = data_path
        self.file_list = [
            f
            for f in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, f)) and f.endswith(".png")
        ]
        self.metadata = pickle.load(open(data_path / "metadata.pkl", "rb"))

    def __len__(self):
        return len(self.file_list)

    def _normalize_label(self, label):
        pos, x_angle, y_angle = [float(l) for l in label]
        # normalize target values
        pos_max = 1324
        pos_min = 0
        pos = (pos - pos_min) / (pos_max - pos_min)
        x_angle_max = 10
        x_angle_min = -10
        x_angle = (x_angle - x_angle_min) / (x_angle_max - x_angle_min)
        y_angle_max = 10
        y_angle_min = -10
        y_angle = (y_angle - y_angle_min) / (y_angle_max - y_angle_min)
        return torch.tensor([pos, x_angle, y_angle])

    def restore_label(self, label):
        pos, x_angle, y_angle = label
        # restore target values
        pos_max = 1324
        pos_min = 0
        pos = pos * (pos_max - pos_min) + pos_min
        x_angle_max = 10
        x_angle_min = -10
        x_angle = x_angle * (x_angle_max - x_angle_min) + x_angle_min
        y_angle_max = 10
        y_angle_min = -10
        y_angle = y_angle * (y_angle_max - y_angle_min) + y_angle_min
        return [pos, x_angle, y_angle]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(
            str(self.data_path / self.file_list[idx]), cv2.IMREAD_GRAYSCALE
        )
        label = self.metadata[self.file_list[idx].split(".")[0]]
        label = [
            label["pos"],
            label["x_angle"],
            label["y_angle"],
        ]
        label = self._normalize_label(label)

        if self.transform:
            image = self.transform(image)

        return image, label


class GaussianNoise:
    """Torch transform that adds Gaussian noise to an image"""

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    from train import Trainer

    model = TissuePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = AngledAtlasDataset(
        Path(r"C:\Users\asoro\Desktop\angled_data"), transform=transforms
    )

    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    trainer = Trainer(
        model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda"
    )
    trainer.run(epochs=300)
