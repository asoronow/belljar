import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import csv
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchsummary import summary
import argparse


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # The fc_loc layers will be defined in the forward method once we know the output size of the localization network
        self.fc_loc = None

    def forward(self, x):
        xs = self.localization(x)
        xs_size = xs.size()
        if self.fc_loc is None:
            self.fc_loc = nn.Sequential(
                nn.Linear(xs_size[1] * xs_size[2] * xs_size[3], 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        xs = xs.view(xs.size(0), -1)  # Flatten dynamically
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class SliceEstimator(nn.Module):
    def __init__(self, block, layers):
        super(SliceEstimator, self).__init__()
        self.in_channels = 64
        self.stn = STN()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=3)

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc_x = nn.Linear(512, 1)
        self.fc_y = nn.Linear(512, 1)
        self.fc_z = nn.Linear(512, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        batch_size, channels, height, width = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        x = self.global_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x_cut = self.fc_x(x)
        y_cut = self.fc_y(x)
        z_depth = self.fc_z(x)

        return x_cut, y_cut, z_depth

layers = [9, 36, 67, 9]

# Create the model instance
model = SliceEstimator(ResidualBlock, layers)


class SytheticSliceDataset(Dataset):
    """
    A custom dataset for the Sythetic Slice Estimator

    Args:
        Dataset (Dataset): A PyTorch Dataset
        metadata_path (str): Path to the metadata file
        images_path (str): Path to the images
    """
    def __init__(self, metadata_path, images_path, transforms=None):
        self.metadata = Path(metadata_path)
        self.images = Path(images_path)
        self.transforms = transforms
        self.data = {}

    def _init_data(self):
        """Reads metadata csv into a dictionary
        CSV row format: file_name, x_angle, y_angle, z_position
        """
        with open(self.metadata, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in reader:
                self.data[row[0]] = {
                    "x_angle": float(row[1]),
                    "y_angle": float(row[2]),
                    "z_position": float(row[3]),
                }

    def _load_image(self, file_name):
        """Loads an image from disk
        Args:
            file_name (str): The filename of the image 
        """
        return cv2.imread(str(self.images / file_name), cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.data)

    def _normalize_label(self, label):
        """Normalizes the label"""
        pos, x_angle, y_angle = [float(l) for l in label]
        pos_max = 1324
        pos_min = 0
        pos = (pos - pos_min) / (pos_max - pos_min)
        x_angle_max = 180
        x_angle_min = -180
        x_angle = (x_angle - x_angle_min) / (x_angle_max - x_angle_min)
        y_angle_max = 180
        y_angle_min = -180
        y_angle = (y_angle - y_angle_min) / (y_angle_max - y_angle_min)
        return torch.tensor([pos, x_angle, y_angle])
    
    def restore_label(self, label):
        pos, x_angle, y_angle = label
        # restore target values
        pos_max = 1324
        pos_min = 0
        pos = pos * (pos_max - pos_min) + pos_min
        x_angle_max = 180
        x_angle_min = -180
        x_angle = x_angle * (x_angle_max - x_angle_min) + x_angle_min
        y_angle_max = 180
        y_angle_min = -180
        y_angle = y_angle * (y_angle_max - y_angle_min) + y_angle_min
        return [pos, x_angle, y_angle]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self._load_image(self.data[idx]["file_name"])
        if transforms is not None:
            image = transforms(image)

        label = [
            self.data[idx]["x_angle"],
            self.data[idx]["y_angle"],
            self.data[idx]["z_position"],
        ]

        label = self._normalize_label(label)

        return image, label



if __name__ == "__main__":
    # args
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--metadata", type=str, required=True)
    args.add_argument("-i", "--images", type=str, required=True)
    args.add_argument("-e", "--epochs", type=int, required=True, default=10)
    args.add_argument("-b", "--batch_size", type=int, required=True, default=8)
    args.add_argument("-l", "--learning_rate", type=float, required=True, default=0.01)
    args = args.parse_args()

    # Setup training loop
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)

    # Create the model instance
    model = SliceEstimator(ResidualBlock, layers)

    # Random contrast and brightness transform
    transforms = transforms.Compose(
        [
            transforms.ColorJitter((0.0, 0.25), (0.0, 0.25)),
            transforms.ToTensor(),
        ]
    )

    # Create the data loaders
    train_dataset = SytheticSliceDataset(args.metadata.strip(), args.images.strip(), transforms)
    
    # Train/Val split
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [0.8, 0.2]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Create the loss function
    criterion = nn.MSELoss()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    for epoch in range(epochs):
        train_loss = 0.0
        for batch, (samples, labels) in enumerate(train_dataloader):
            print(
                f"Train | Epoch: {epoch}, Batch: {batch}/{len(train_dataloader)}",
                end="\r",
            )
            samples = samples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * samples.size(0)
        print(f"\nTrain Loss: {train_loss / len(train_dataloader.dataset)}")

        valid_loss = 0.0
        with torch.no_grad():
            for batch, (samples, labels) in enumerate(val_dataloader):
                print(
                    f"Valid | Epoch: {epoch}, Batch: {batch}/{len(val_dataloader)}",
                    end="\r",
                )
                samples = samples.to(device)
                labels = labels.to(device)
                outputs = model(samples)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * samples.size(0)
            if valid_loss < best_loss:
                best_loss = valid_loss
                # Save
                torch.save(model.state_dict(), f"best_model_{epoch}.pt")
        print(f"\nValid Loss: {valid_loss / len(val_dataloader.dataset)}")

        



