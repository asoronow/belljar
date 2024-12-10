import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from pathlib import Path
import argparse
import numpy as np
from pytorch_msssim import SSIM
import cv2
import os
import shutil
import wandb
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
import torch.distributed as dist
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class BrainRegUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BrainRegUNet, self).__init__()

        self.encoder1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder4 = self._conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder3 = self._conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder2 = self._conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.decoder1 = self._conv_block(64, 32)

        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.crop_and_concat(enc4, dec4)
        att4 = self.att4(dec4, enc4)
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.crop_and_concat(enc3, dec3)
        att3 = self.att3(dec3, enc3)
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.crop_and_concat(enc2, dec2)
        att2 = self.att2(dec2, enc2)
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.crop_and_concat(enc1, dec1)
        att1 = self.att1(dec1, enc1)
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def crop_and_concat(self, upsampled, bypass):
        crop_size = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-crop_size, -crop_size, -crop_size, -crop_size))
        return bypass
    
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
    def __init__(self, originals, targets, transform=None, target_transform=None, original_transform=None):
        self.originals = originals
        self.targets = targets

        # Ensure the same number of images
        assert len(self.originals) == len(self.targets)

        self.transform = transform
        self.target_transform = target_transform
        self.original_transform = original_transform

    def sobel_edge_detection(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        return sobel_combined.astype(np.uint8)

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        original = self.originals[idx]
        target = self.targets[idx]

        original = cv2.imread(str(original), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)

        original = self.sobel_edge_detection(original)
        target = self.sobel_edge_detection(target)
      
        if self.transform:
            original = self.transform(original)
            target = self.transform(target)

        if self.target_transform:
            target = self.target_transform(target)


        if self.original_transform:
            original = self.original_transform(original)           

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
            transforms.ToTensor(),
        ]
    )

    original_transform = transforms.Compose(
        [
            transforms.RandomRotation(30),
        ]
    )        

    def get_run_count(directory):
        runs = [int(x.stem.split('run')[1]) for x in Path(directory).iterdir() if x.stem.startswith('run')]
        return max(runs) + 1 if runs else 0

    def create_run_folder(directory):
        run_count = get_run_count(directory)
        run_path = Path(directory) / f'run{run_count}'
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    if rank == 0:
        if os.path.exists('/workspace'):
            run_path = create_run_folder('/workspace')

    dataset = PairedDataset(originals, targets, transform=transform, original_transform=original_transform)
    # limit dataset to 1000 images for quick testing on local
    if world_size == 1:
        dataset = torch.utils.data.Subset(dataset, range(1000))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = BrainRegUNet(in_channels=2, out_channels=2).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

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
        # save current model
        if rank == 0:
            torch.save(model.state_dict(), 'last_brain_reg_net.pt')
            if os.path.exists('/workspace'):
                shutil.copy('last_brain_reg_net.pt', run_path / 'last_brain_reg_net.pt')
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if rank == 0:  # Only save on rank 0 to avoid overwriting       
                torch.save(model.module.state_dict(), 'best_brain_reg_net.pt')
                if os.path.exists('/workspace'):
                    shutil.copy('best_brain_reg_net.pt', run_path / 'best_brain_reg_net.pt')
                # display_images(original[0], target[0], warped_original[0])

        # Log loss
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

def fine_tune_model(args):
    originals_path = args.originals_path
    targets_path = args.targets_path
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainRegUNet(in_channels=2, out_channels=2).to(device)
    # load state dict sensitve to module
    model.load_state_dict(remove_module_prefix(torch.load('weights_brain_reg_net.pt')))
    # Load real-world data
    originals_path = Path(originals_path).expanduser()
    targets_path = Path(targets_path).expanduser()

    originals = sorted(originals_path.glob("*.png"))
    targets = sorted(targets_path.glob("*.png"))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match network input size
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.25, 0.25,),  # Add color jitter to match network
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    full_dataset = PairedDataset(originals, targets, transform=transform, target_transform=target_transform)
    # Select a subset of real-world data for fine-tuning
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Freeze all layers except the last few
    for param in model.parameters():
        param.requires_grad = False
    
    unfreeze = [
        model.conv,
        model.decoder1,
        model.decoder2,
        model.decoder3,
        model.decoder4,
        model.att1,
        model.att2,
        model.att3,
        model.att4,
        model.upconv1,
        model.upconv2,
        model.upconv3,
        model.upconv4
    ]

    for layer in unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
 


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=1).to(device)
    
    model.train()

    best_loss = float("inf")
    patience_step = 0
    last_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for original, target in dataloader:
            original = original.to(device)
            target = target.to(device)
            input_pair = torch.cat([original, target], dim=1)

            deformation_field = model(input_pair)
            grid = create_grid(original.size(0), original.shape[2:]).to(device)
            warped_grid = grid + deformation_field.permute(0, 2, 3, 1)
            warped_original = F.grid_sample(original, warped_grid, align_corners=True)

            similarity_loss = ssim_loss(warped_original, target)
            smooth_loss = smoothness_loss(deformation_field)
            loss = similarity_loss + smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * original.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'fine_tuned_brain_reg_net.pt')

        # Early stopping
        last_losses.append(epoch_loss)
        num_losses_to_track = 10
        if len(last_losses) > num_losses_to_track:
            last_losses = last_losses[-num_losses_to_track:]
        if len(last_losses) >= num_losses_to_track:
            losses_decreasing = all(loss < last_losses[i] for i, loss in enumerate(last_losses[:-1]))
            if losses_decreasing:
                patience_step += 1
                if patience_step >= 20:
                    print("Early stopping!")
                    break
            else:
                patience_step = 0

    return model

def test(args):
    originals_path = Path(args.originals_path).expanduser()
    targets_path = Path(args.targets_path).expanduser()
    originals = sorted(originals_path.glob("*.png"))
    targets = sorted(targets_path.glob("*.png"))

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = PairedDataset(originals, targets, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    model = BrainRegUNet(in_channels=2, out_channels=2).to('cuda')
    ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=1).to('cuda')
    # load state dict sensitve to module
    model.load_state_dict(remove_module_prefix(torch.load('weights_brain_reg_net.pt')))
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

            for i in range(batch_size):
                # get the ssim loss
                similarity_loss = ssim_loss(warped_original[i].unsqueeze(0), target[i].unsqueeze(0))
                print(f"SSIM Loss: {similarity_loss:.4f}")
                display_images(original[i], target[i], warped_original[i])

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

    # fine_tune_model(args)
    # test(args)
    world_size = args.world_size
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
