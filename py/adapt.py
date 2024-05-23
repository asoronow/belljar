import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps
from model import UNet, UNetWithGRL
from torchvision import transforms
import os
import numpy as np
from train import AtlasDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class DAPIDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert("L")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        if self.transform:
            image = self.transform(image)
            # normalize
            image = transforms.Normalize((20.59695 / 255,), (40.319914 / 255,))(image)
        return image


class DomainDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(DomainDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Initialize the adaptable UNet with GRL
unet = UNetWithGRL(1, 1328)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_weights = torch.load("unet_best_labler.pt")
unet.load_state_dict(pretrained_weights)
unet.to(device)
# Initialize the Domain Classifier
domain_classifier = DomainDiscriminator(128)
domain_classifier.to(device)
# Define the loss functions
segmentation_loss = nn.CrossEntropyLoss()
domain_loss = nn.BCELoss()

# Define the optimizers
unet_optimizer = optim.SGD(unet.parameters(), lr=0.01, momentum=0.9)
domain_optimizer = optim.SGD(domain_classifier.parameters(), lr=0.1, momentum=0.9)


# Training loop
def train_dann(source_dataloader, target_dataloader, num_epochs):
    for epoch in range(num_epochs):
        total_domain_loss = 0.0
        total_seg_loss = 0.0
        print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
        for i, (source_images, source_labels) in enumerate(source_dataloader):
            # Perform a forward pass on source samples
            source_predictions = unet(source_images.to(device), alpha=2.0)
            source_labels = (
                nn.functional.one_hot(source_labels.long(), num_classes=1328)
                .permute(0, 4, 1, 2, 3)
                .reshape([source_labels.shape[0], 1328, 256, 256])
                .float()
            ).to(device)
            # Compute the segmentation loss for source samples
            seg_loss = segmentation_loss(source_predictions, source_labels)

            # Generate domain labels (1 for source, 0 for target)
            source_domain_labels = torch.ones(source_predictions.size(0)).to(device)
            target_images = next(iter(target_dataloader))
            target_domain_labels = torch.zeros(target_images.size(0)).to(device)

            # Perform a forward pass on both source and target samples
            target_features = unet.get_features(target_images.to(device))
            source_features = unet.get_features(source_images.to(device))

            # Compute the domain predictions
            source_domain_predictions = domain_classifier(source_features[1].detach())
            target_domain_predictions = domain_classifier(target_features[1].detach())
            source_domain_labels = source_domain_labels.unsqueeze(1)
            target_domain_labels = target_domain_labels.unsqueeze(1)
            # Compute the domain loss
            domain_loss_value = domain_loss(
                source_domain_predictions, source_domain_labels
            ) + domain_loss(target_domain_predictions, target_domain_labels)
            total_domain_loss += domain_loss_value.item() * source_images.size(0)
            total_seg_loss += seg_loss.item() * source_images.size(0)
            print(
                f"Domain Loss: {total_domain_loss / len(source_dataloader.dataset + target_dataloader.dataset):.4f} | Seg Loss: {total_seg_loss / len(source_dataloader.dataset):.4f}",
                end="\r",
            )
            # Compute the total loss as a combination of segmentation and domain adaptation losses
            total_loss = seg_loss + 1.0 * domain_loss_value

            # Backpropagate and update the UNet based on the total loss
            unet_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            unet_optimizer.step()

            # Show the new map of the target image
            # target_predictions = unet(target_images.to(device), alpha=0.0)
            # target_predictions = target_predictions.argmax(1)
            # target_predictions = target_predictions.cpu().detach().numpy()
            # target_predictions = np.squeeze(target_predictions)
            # target_predictions = target_predictions[0, :, :]
            # blank = np.zeros((256, 256, 3))
            # for i in range(1328):
            #     blank[target_predictions == i] = source_dataset.classMap[
            #         source_dataset.classToKey(i)
            #     ]["color"]
            # blank = Image.fromarray(blank.astype(np.uint8))
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(target_images[0, 0, :, :], cmap="gray")
            # ax[1].imshow(blank)
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()

            # Backpropagate and update the Domain Classifier based on the domain loss
            domain_optimizer.zero_grad()
            domain_loss_value.backward()
            domain_optimizer.step()

        torch.save(unet.state_dict(), "adapted_unet_weights.pth")
        torch.save(domain_classifier.state_dict(), "domain_classifier_weights.pth")


imagePath = "C:\\Users\\Alec\\Projects\\aba-nrrd\\image"
mapPath = "C:\\Users\\Alec\\Projects\\aba-nrrd\\map"
dapiPath = "C:\\Users\\Alec\\Downloads\\DAPI"

image_paths = []
for root, dirs, files in os.walk(imagePath):
    for file in files:
        if file.endswith(".png"):
            image_paths.append(os.path.join(root, file))

# walk through the map directory and get all the map paths
map_paths = []
for root, dirs, files in os.walk(mapPath):
    for file in files:
        if file.endswith(".tif"):
            map_paths.append(os.path.join(root, file))

dapi_paths = []
for root, dirs, files in os.walk(dapiPath):
    for file in files:
        if file.endswith(".png"):
            dapi_paths.append(os.path.join(root, file))

source_dataset = AtlasDataset(
    image_paths,
    map_paths,
    transform=transforms.ToTensor(),
    random_affine=False,
)
target_dataset = DAPIDataset(dapi_paths, transform=transforms.ToTensor())
source_dataloader = DataLoader(source_dataset, batch_size=6, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=6, shuffle=True)

train_dann(source_dataloader, target_dataloader, num_epochs=10)
