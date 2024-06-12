import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import csv
import cv2
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from pathlib import Path
import argparse
import os
import logging
import sys
import wandb


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
        
        # Initialize fc_loc layers but leave them without fixed input size
        self.fc_loc1 = None
        self.fc_loc2 = None

    def forward(self, x):
        xs = self.localization(x)
        xs_size = xs.size()
        
        if self.fc_loc1 is None or self.fc_loc2 is None:
            # Compute the size for the fc_loc1 layer dynamically
            flatten_size = xs_size[1] * xs_size[2] * xs_size[3]
            self.fc_loc1 = nn.Linear(flatten_size, 32).to(xs.device)
            self.fc_loc2 = nn.Linear(32, 6).to(xs.device)
            # Initialize the weights/bias with identity transformation
            self.fc_loc2.weight.data.zero_()
            self.fc_loc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        xs = xs.view(xs.size(0), -1)  # Flatten dynamically
        xs = self.fc_loc1(xs)
        xs = F.relu(xs)
        theta = self.fc_loc2(xs)
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
        out = torch.stack([z_depth, x_cut, y_cut], dim=1)
        return out

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
    def __init__(self, metadata_path, images_path, transform=None):
        self.metadata = Path(metadata_path)
        self.images = Path(images_path)
        self.transform = transform
        self.data = []
 
        with open(self.metadata, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            # Skip the header
            next(reader)
            for i, row in enumerate(reader):
                self.data.append({
                    "file_name": row[0],
                    "x_angle": float(row[1]),
                    "y_angle": float(row[2]),
                    "z_position": float(row[3]),
                })

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
        if self.transform is not None:
            image = self.transform(image)

        label = [
            self.data[idx]["x_angle"],
            self.data[idx]["y_angle"],
            self.data[idx]["z_position"],
        ]

        label = self._normalize_label(label)

        return image, label
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # check os, if windows, set to gloo
    if os.name == 'nt':
        backend = 'gloo'
    else:
        backend = 'nccl'
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log_{rank}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    if rank == 0:
        wandb.init(project="slice_estimation")

    # Set the device
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Create the model instance and move to device
    model = SliceEstimator(ResidualBlock, layers).to(device)
    model = DDP(model, device_ids=[rank])

    # Random contrast and brightness transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter((0.0, 0.25), (0.0, 0.25)),
        transforms.ToTensor(),
    ])

    # Create the dataset and the distributed data loaders
    og_dataset = SytheticSliceDataset(args.metadata.strip(), args.images.strip(), transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(og_dataset, [0.8, 0.2])

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # Create the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_sampler.set_epoch(epoch)
        for batch, (samples, labels) in enumerate(train_dataloader):
            samples = samples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * samples.size(0)
            if rank == 0:
                logging.info(f"Rank {rank} | Epoch {epoch} | Batch {batch + 1} / {len(train_dataloader)} | Train Loss: {train_loss / ((batch + 1) * args.batch_size)}")
        
        train_loss = train_loss / len(train_dataset)
        if rank == 0:
            wandb.log({"train_loss": train_loss})

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch, (samples, labels) in enumerate(val_dataloader):
                samples = samples.to(device)
                labels = labels.to(device)
                outputs = model(samples)
                loss = criterion(outputs.squeeze(), labels)
                valid_loss += loss.item() * samples.size(0)
                if rank == 0:
                    logging.info(f"Rank {rank} | Epoch {epoch} | Batch {batch} / {len(val_dataloader)} | Valid Loss: {valid_loss / ((batch + 1) * args.batch_size)}")

            valid_loss = valid_loss / len(val_dataset)
            if valid_loss < best_loss:
                best_loss = valid_loss
                if rank == 0:
                    torch.save(model.state_dict(), f"best.pt")
            
            if rank == 0:
                torch.save(model.state_dict(), f"last.pt")
                wandb.log({"best_loss": best_loss})
                wandb.log({"valid_loss": valid_loss})

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata", type=str, required=True)
    parser.add_argument("-i", "--images", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=False, default=10)
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=16)
    parser.add_argument("-l", "--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument("-n", "--nodes", type=int, default=1, help="number of nodes for distributed training")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="number of gpus per node")
    parser.add_argument("-nr", "--nr", type=int, default=0, help="ranking within the nodes")
    args = parser.parse_args()

    world_size = args.gpus * args.nodes
    mp.spawn(train, args=(world_size, args), nprocs=args.gpus, join=True)