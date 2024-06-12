import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from pathlib import Path
import argparse
from pytorch_msssim import SSIM
import cv2
import os
import wandb
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
import torch.distributed as dist

class BrainRegNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, init_features=32):
        super(BrainRegNet, self).__init__()

        features = init_features
        self.encoder1 = BrainRegNet._block(in_channels, features, name="enc1", dropout=0.3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = BrainRegNet._block(features, features * 2, name="enc2", dropout=0.3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = BrainRegNet._block(features * 2, features * 4, name="enc3", dropout=0.3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = BrainRegNet._block(features * 4, features * 8, name="enc4", dropout=0.3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = BrainRegNet._block(features * 8, features * 16, name="bottleneck", dropout=0.3)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = BrainRegNet._block((features * 8) * 2, features * 8, name="dec4", dropout=0.3)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = BrainRegNet._block((features * 4) * 2, features * 4, name="dec3", dropout=0.3)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = BrainRegNet._block((features * 2) * 2, features * 2, name="dec2", dropout=0.3)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = BrainRegNet._block(features * 2, features, name="dec1", dropout=0.3)
        
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )



class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)

def smoothness_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)

def create_grid(batch_size, shape):
    N, H, W = batch_size, shape[0], shape[1]
    tensors = [torch.linspace(-1, 1, s) for s in [H, W]]
    grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    return grid

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class PairedDataset(Dataset):
    def __init__(self, originals, targets, transform=None):
        self.originals = originals
        self.targets = targets

        # Ensure the same number of images
        assert len(self.originals) == len(self.targets)

        self.transform = transform

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        original = self.originals[idx]
        target = self.targets[idx]

        original = cv2.imread(str(original), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            original = self.transform(original)
            target = self.transform(target)

        return original, target

def display_images(original, target, warped):
    original = original.squeeze().cpu().numpy()
    warped = warped.squeeze().cpu().detach().numpy()
    target = target.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(warped, cmap='gray')
    axes[1].set_title('Warped Image')
    axes[2].imshow(target, cmap='gray')
    axes[2].set_title('Target Image')
    plt.show()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if os.name == 'nt':
        backend = 'gloo'
    else:
        backend = 'nccl'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project="deep_registration")
    
    originals_path = Path(args.originals_path).expanduser()
    targets_path = Path(args.targets_path).expanduser()

    originals = sorted(originals_path.glob("*.png"))
    targets = sorted(targets_path.glob("*.png"))

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=[0.0,0.25], contrast=[0.0,0.25]),
            transforms.ToTensor(),
        ]
    )

    dataset = PairedDataset(originals, targets, transform=transform)
    # limit dataset to 1000 images for quick testing
    # dataset = torch.utils.data.Subset(dataset, range(10000))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = BrainRegNet(in_channels=2, out_channels=2, init_features=32).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=1).to(rank)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_samples = 0
        sampler.set_epoch(epoch)
        for batch, (original, target) in enumerate(dataloader):
            original = original.to(rank)
            target = target.to(rank)

            input_pair = torch.cat([original, target], dim=1)  # Concatenate along channel dimension
            deformation_field = model(input_pair) 

            batch_size = original.size(0)
            num_samples += batch_size
            grid = create_grid(batch_size, original.shape[2:]).to(rank)
            warped_grid = grid + deformation_field.permute(0, 2, 3, 1)
            warped_original = F.grid_sample(original, warped_grid, align_corners=True)

            similarity_loss = ssim_loss(warped_original, target)
            smooth_loss = smoothness_loss(deformation_field)
            loss = similarity_loss + smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            if rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch + 1} / {len(dataloader)}, Loss: {train_loss / num_samples:.4f}")

        epoch_loss = train_loss / num_samples

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if rank == 0:  # Only save on rank 0 to avoid overwriting
                torch.save(model.module.state_dict(), 'brain_reg_net.pt')
                # display_images(original[0], target[0], warped_original[0])
        # Display the first image at the end of each epoch

        if rank == 0:
            wandb.log({"loss": epoch_loss})
    
    cleanup()

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def test(args):
    originals_path = Path(args.originals_path).expanduser()
    targets_path = Path(args.targets_path).expanduser()
    originals = sorted(originals_path.glob("*.png"))
    targets = sorted(targets_path.glob("*.png"))

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),  # Ensure images are single-channel
            transforms.ToTensor(),
        ]
    )

    dataset = PairedDataset(originals, targets, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    model = BrainRegNet(in_channels=2, out_channels=2, init_features=32).to('cuda')
    # load state dict sensitve to module
    model.load_state_dict(remove_module_prefix(torch.load('brain_reg_net.pt')))
    model.eval()

    with torch.no_grad():
        for original, target in dataloader:
            original = original.to('cuda')
            target = target.to('cuda')

            input_pair = torch.cat([original, target], dim=1)  # Concatenate along channel dimension
            deformation_field = model(input_pair) 

            batch_size = original.size(0)
            grid = create_grid(batch_size, original.shape[2:]).to('cuda')
            warped_grid = grid + deformation_field.permute(0, 2, 3, 1)
            warped_original = F.grid_sample(original, warped_grid, align_corners=True)

            display_images(original[0], target[0], warped_original[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("originals_path", type=str)
    parser.add_argument("targets_path", type=str)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()


    #test(args)
    world_size = args.world_size
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
