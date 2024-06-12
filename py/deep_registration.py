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
from timm import create_model


class BrainRegViT(nn.Module):
    def __init__(self, backbone='vit_base_patch16_224', img_size=224, num_classes=2, init_features=64, dropout_rate=0.3):
        super(BrainRegViT, self).__init__()

        # Vision Transformer backbone
        self.backbone = create_model(backbone, pretrained=True, img_size=img_size)
         # Calculate the correct input size for the head network
        vit_embed_dim = self.backbone.embed_dim
        num_patches = (img_size // 16) ** 2 + 1  # +1 for class token
        input_dim = vit_embed_dim * num_patches * 2
        # Deeper head network
        self.head = nn.Sequential(
            nn.Linear(input_dim, init_features),  # *2 for concatenated embeddings
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(init_features, init_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(init_features // 2, init_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(init_features // 4, img_size * img_size * num_classes)  # Predict deformation field
        )
        self.img_size = img_size
        self.num_classes = num_classes

    def forward(self, original, target):
        # Extract features from original and target images
        original_features = self.backbone.forward_features(original)
        target_features = self.backbone.forward_features(target)
        # Concatenate the features
        features = torch.cat((original_features, target_features), dim=-1)
        features = features.view(features.size(0), -1)
        # Predict deformation field
        x = self.head(features)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_classes, self.img_size, self.img_size)
        return x


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(SSIM_Loss, self).forward(img1, img2))

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

        original = cv2.imread(str(original), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target), cv2.IMREAD_COLOR)
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
    # dataset = torch.utils.data.Subset(dataset, range(1000))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = BrainRegViT().to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=3).to(rank)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_samples = 0
        sampler.set_epoch(epoch)
        for batch, (original, target) in enumerate(dataloader):
            original = original.to(rank)
            target = target.to(rank)

            deformation_field = model(original, target) 
            batch_size = original.size(0)
            num_samples += batch_size
            grid = create_grid(batch_size, original.shape[2:]).to(rank)
            warped_grid = grid + deformation_field.permute(0, 2, 3, 1)
            warped_original = F.grid_sample(original, warped_grid, align_corners=True)

            # Compute loss
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
            transforms.ToTensor(),
        ]
    )

    dataset = PairedDataset(originals, targets, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    model = BrainRegViT().to('cuda')
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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("originals_path", type=str)
    parser.add_argument("targets_path", type=str)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()


    #test(args)
    world_size = args.world_size
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
