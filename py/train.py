import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from model import UNet_3Plus
import sys, os, pickle
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from torch.backends import cudnn

cudnn.benchmark = True


def progress_bar(
    current, total, epoch, num_epochs, train_loss, valid_loss=False, bar_length=40
):
    # Format progress bar string
    if not valid_loss:
        progress_str = f"Epoch [{epoch+1}/{num_epochs}], Step [{current+1}/{total}], Train Loss: {train_loss:.4f}"
    else:
        progress_str = f"Epoch [{epoch+1}/{num_epochs}], Step [{current+1}/{total}], Train Loss: {train_loss :.4f}, Valid Loss: {valid_loss:.4f}"
    bar_width = int(bar_length * (current + 1) / total)
    bar_str = "[" + "=" * bar_width + " " * (bar_length - bar_width) + "]"
    progress_str += " " + bar_str
    # Print progress bar string and move cursor up one line
    sys.stdout.write("\r" + progress_str)
    sys.stdout.flush()
    if current == total - 1:
        sys.stdout.write("\n")
        sys.stdout.flush()


class Trainer:
    """Trains a model with the provided dataset and criterion"""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        num_epochs,
        device,
        checkpoint_path,
        wandb_project_name,
        wandb_run_name,
        model_name="best.pt",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.model_name = model_name

        self.best_valid_loss = float("inf")
        self.train_losses = []
        self.valid_losses = []
        self.wandb = wandb

        self.wandb.init(project=self.wandb_project_name, name=self.wandb_run_name)
        self.wandb.watch(self.model)

    def train(self):
        """Trains the model for num_epochs epochs and saves the best model"""
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validate_epoch(epoch, train_loss)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint(self.model_name)
        self.wandb.finish()

    def train_epoch(self, epoch):
        """Trains the model for one epoch"""
        self.model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            progress_bar(
                i,
                len(self.train_loader),
                epoch,
                self.num_epochs,
                train_loss / (i + 1),
            )
        return train_loss / len(self.train_loader)

    def validate_epoch(self, epoch, train_loss):
        """Validates the model for one epoch"""
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.valid_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()
                progress_bar(
                    i,
                    len(self.valid_loader),
                    epoch,
                    self.num_epochs,
                    train_loss,
                    valid_loss / (i + 1),
                )
        return valid_loss / len(self.valid_loader)

    def test(self, test_loader, classMap):
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if labels.max() == 0:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.cpu().squeeze().numpy()
                labels = labels.cpu().squeeze().numpy()
                output_color = np.zeros((outputs.shape[0], outputs.shape[1], 3))
                label_color = np.zeros((labels.shape[0], labels.shape[1], 3))
                index_to_key = {v["index"]: k for k, v in classMap.items()}
                for j in range(outputs.shape[0]):
                    for k in range(outputs.shape[1]):
                        # get key where index is value
                        output_color[j, k, :] = classMap[index_to_key[outputs[j, k]]][
                            "color"
                        ]
                        label_color[j, k, :] = classMap[index_to_key[labels[j, k]]][
                            "color"
                        ]

                image = images.cpu().numpy().squeeze()
                output_color = output_color.astype(np.uint8)
                label_color = label_color.astype(np.uint8)
                ax = plt.subplot(1, 3, 1)
                ax.imshow(output_color)
                ax = plt.subplot(1, 3, 2)
                ax.imshow(label_color)
                ax = plt.subplot(1, 3, 3)
                ax.imshow(image)
                plt.show()

    def save_checkpoint(self, name):
        """Saves the model state dictionary to a file"""
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, name))


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AtlasDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        transform=None,
        random_affine=False,
        classMapPath="classMap.pkl",
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.usedColors = []
        self.random_affine = random_affine
        # get class definitions
        self.classMap = {}
        with open(classMapPath, "rb") as handle:
            self.classMap = pickle.load(handle)

    def __len__(self):
        return len(self.image_paths)

    def classToKey(self, classIndex):
        for k, v in self.classMap.items():
            if v["index"] == classIndex:
                return k

    def same_random_affine_transform_pil(
        self, images, rotation_range=180, shear_range=0.1
    ):
        # pick parameters
        angle = np.random.uniform(-rotation_range, rotation_range)
        shear = np.random.uniform(-shear_range, shear_range)

        # apply transformation
        transformed_images = []
        for image in images:
            transformed_image = image.transform(
                image.size,
                Image.AFFINE,
                (1, shear, 0, shear, 1, 0),
                Image.Resampling.NEAREST,
                fillcolor=0,
            )
            transformed_image = transformed_image.rotate(
                angle, Image.Resampling.NEAREST
            )
            transformed_image = transformed_image.resize(
                image.size, Image.Resampling.NEAREST
            )

            transformed_images.append(transformed_image)

        return transformed_images

    def __getitem__(self, index):
        # Load image and label
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])

        if image.mode != "L":
            image = image.convert("L")
        # Apply the same transformation to image and label
        if self.random_affine:
            image, label = self.same_random_affine_transform_pil([image, label])

        # Convert label image to classes
        label = np.array(label)
        class_indices = np.vectorize(lambda x: self.classMap[np.int32(x)]["index"])
        label = class_indices(label)

        # Apply any additional transformations (e.g., normalization)
        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                AddGaussianNoise(0.0, 0.001),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if self.transform:
            image = image_transforms(image)
            label = self.transform(label).squeeze().long()

        # Return the resized image and label
        return image, label


def gather_paths(root, extension=None, filter=None):
    all_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if extension is None or name.endswith(extension):
                if filter is None or filter in name:
                    all_files.append(os.path.join(path, name))

    return all_files


def unique_noun_verb_name():
    animals = [
        "Aardvark",
        "Albatross",
        "Alligator",
        "Alpaca",
        "Ant",
        "Anteater",
        "Antelope",
        "Ape",
        "Armadillo",
        "Donkey",
        "Baboon",
        "Badger",
        "Barracuda",
        "Bat",
        "Bear",
        "Beaver",
        "Bee",
        "Bison",
        "Boar",
        "Buffalo",
        "Butterfly",
    ]

    verbs = [
        "Abduct",
        "Abhor",
        "Abide",
        "Accelerate",
        "Running",
        "Leaping",
        "Jumping",
        "Crawling",
        "Climbing",
        "Swimming",
        "Flying",
    ]

    places = [
        "Abyss",
        "Trench",
        "Cave",
        "Forest",
        "Jungle",
        "Desert",
        "Mountain",
        "Valley",
        "River",
        "Ocean",
        "Sea",
        "Lake",
        "Pond",
    ]

    return (
        np.random.choice(animals)
        + np.random.choice(verbs)
        + np.random.choice(places)
        + str(np.random.randint(0, 1000 + 1))
    )


if __name__ == "__main__":
    # Get our images and labels
    atlasImagesPath = "C:\\Users\\Alec\\Downloads\\M511-Alignment\\"
    atlasLabelsPath = "C:\\Users\\Alec\\Downloads\\M511-Alignment\\"
    atlasImages = gather_paths(atlasImagesPath, ".png")
    atlasLabels = gather_paths(atlasLabelsPath, ".tif")

    print(len(atlasImages), len(atlasLabels))

    random_indices = np.random.choice(len(atlasImages), 100)
    # Make the dataset
    dataset = AtlasDataset(
        atlasImages,
        atlasLabels,
        transform=transforms.ToTensor(),
        random_affine=False,
        classMapPath="C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\classMap.pkl",
    )

    # Split into train and test
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Define the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Define the model
    model = UNet_3Plus(1, 673)
    model.load_state_dict(
        torch.load("C:\\Users\\Alec\\Projects\\aba-nrrd\\models\\best-dapi.pt")
    )
    model = model.cuda()
    # Define the loss function
    loss = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    trainer = Trainer(
        model,
        loss,
        optimizer,
        train_loader,
        test_loader,
        50,
        device="cuda:0",
        checkpoint_path="C:\\Users\\Alec\\Projects\\aba-nrrd\\models",
        wandb_project_name="unet3plus_alignment",
        wandb_run_name=unique_noun_verb_name(),
        model_name="best-dapi.pt",
    )

    dapiImage = gather_paths("C:\\Users\\Alec\\Downloads\\DAPI", ".png")
    dapiDataset = AtlasDataset(
        dapiImage,
        atlasLabels,
        transform=transforms.ToTensor(),
        random_affine=False,
        classMapPath="C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\classMap.pkl",
    )
    test_loader_single = DataLoader(
        dapiDataset, batch_size=1, shuffle=False, num_workers=4
    )
    trainer.test(test_loader_single, dataset.classMap)
    trainer.train()
