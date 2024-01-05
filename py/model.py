import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Function
from pathlib import Path
import os, itertools
import pickle
import cv2
from PIL import Image
import numpy as np

class ResidualBlock(nn.Module):
    '''
    Residual Block for ResNet architecture
    '''
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class TissuePredictor(nn.Module):
    '''
    TissuePredictor is a ResNet-18 based model for predicting tissue type from a histology image
    '''
    def __init__(self):
        super(TissuePredictor, self).__init__()

        # Initial Convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten

        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x

    def extract_features(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        features = torch.flatten(x, 1)

        return features

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

        if transform:
            self.transform = transform

    def sobel(self, image):
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
        gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
        gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)
        
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)

        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        return combined

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)
        if image.size != (256, 256):
            image = image.resize((256, 256))
        image = np.array(image)
        image = self.sobel(image)
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
    
    def sobel(self, image):
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
        gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
        gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)
        
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)

        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        return combined
    
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
        image = self.sobel(image)
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

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANNClassifier(nn.Module):
    def __init__(self):
        super(DANNClassifier, self).__init__()
        # Domain classifier layers
        self.domain_classifier = nn.Sequential(
            nn.Linear(256**2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2)  # Binary classification: synthetic or real
        )

    def forward(self, x, alpha):
        reversed_features = ReverseLayerF.apply(x, alpha)
        domain_output = self.domain_classifier(reversed_features)
        domain_output = F.softmax(domain_output, dim=1)
        return domain_output


def domain_adaptation():
    import wandb
    def update_hyperparameters(alpha, lambda_weight, source_val_loss, target_val_loss, 
                            alpha_increment=0.05, lambda_adjust_factor=0.1, update_threshold=0.1,
                            max_alpha=1.0):
        """
        Update alpha and lambda_weight based on validation losses.

        Args:
        - alpha (float): Current value of alpha.
        - lambda_weight (float): Current value of lambda_weight.
        - source_val_loss (float): Validation loss on the source domain.
        - target_val_loss (float): Validation loss on the target domain.
        - alpha_increment (float): Amount to increment alpha.
        - lambda_adjust_factor (float): Factor to adjust lambda_weight.
        - update_threshold (float): Minimum relative difference between source and target 
                                    validation losses required to update lambda_weight and alpha.
        - max_alpha (float): Maximum allowable value for alpha.

        Returns:
        - (float, float): Updated values of alpha and lambda_weight.
        """

        # Calculate the relative difference between source and target validation losses
        relative_diff = abs(target_val_loss - source_val_loss) / source_val_loss

        # Update alpha if relative difference exceeds the threshold and alpha is not at its max
        if relative_diff > update_threshold and alpha < max_alpha:
            alpha = min(alpha + alpha_increment, max_alpha)

        # Update lambda_weight based on the relative difference
        if relative_diff > update_threshold:
            if target_val_loss > source_val_loss:
                lambda_weight += lambda_adjust_factor
            else:
                lambda_weight -= lambda_adjust_factor

            # Ensure lambda_weight stays within reasonable bounds
            lambda_weight = max(min(lambda_weight, 1), 0)

        return alpha, lambda_weight



    model = TissuePredictor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_classifier = DANNClassifier()
    model = model.to(device)
    # load pretrained weights
    model.load_state_dict(torch.load("best_model_predictor.pt"))
    domain_classifier = domain_classifier.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    domain_optimizer = torch.optim.AdamW(domain_classifier.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    txs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.125378, std=0.087051),
        ]
    )

    source_dataset = AngledAtlasDataset(
        Path(r"C:\Users\asoro\Desktop\angled_data"), transform=txs
    )
    target_dataset = AtlasDataset(
       Path (r"C:\Users\asoro\Projects\belljar-figures\downloaded_ish_images"), transform=txs
    )
    train_size = int(0.75 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        source_dataset, [train_size, val_size]
    )

    target_train_size = int(0.75 * len(target_dataset))
    target_val_size = len(target_dataset) - target_train_size
    target_train_dataset, target_val_dataset = torch.utils.data.random_split(
        target_dataset, [target_train_size, target_val_size]
    )

    target_train_dataloader = DataLoader(target_train_dataset, batch_size=64, shuffle=True)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=64, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_epochs = 200
    alpha = 0.05
    lambda_weight = 0.1
    best_loss = float("inf")
    wandb.init(project="tissue_dann")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  # Set the model to training mode
        train_source_loss = 0.0
        train_target_loss = 0.0
        for i, ((source_data, source_labels), target_data) in enumerate(zip(train_dataloader, target_train_dataloader)):
            print(f"Train | Epoch: {epoch+1}/{num_epochs} Batch: {i+1}/{len(target_train_dataloader)}", end="\r")
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            target_data = target_data.to(device)

           # Forward pass for source data
            source_pred = model(source_data)
            loss_main = criterion(source_pred, source_labels)

            # Forward pass for domain classification
            combined_features = torch.cat((model.extract_features(source_data),
                                        model.extract_features(target_data)), 0)
            domain_labels = torch.cat((torch.zeros(source_data.size(0)),
                                    torch.ones(target_data.size(0))), 0).long().to(device)
            
            domain_pred = domain_classifier(combined_features, alpha)
            loss_domain = domain_criterion(domain_pred, domain_labels)

            train_source_loss += loss_main.item()
            train_target_loss += loss_domain.item()
            # Compute combined loss
            loss_combined = loss_main + lambda_weight * loss_domain
            

            # Backpropagation and optimization
            optimizer.zero_grad()
            domain_optimizer.zero_grad()
            loss_combined.backward()
            optimizer.step()
            domain_optimizer.step()
        # Validation Phase
        train_source_loss /= len(target_train_dataloader)
        train_target_loss /= len(target_train_dataloader)
        print(f"\nTrain Source Loss: {train_source_loss}, Train Target Loss: {train_target_loss}")
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            b = 0
            # Validation on Source Domain
            source_val_loss = 0.0
            for source_data, source_labels in val_dataloader:
                print(f"Valid | Epoch: {epoch+1}/{num_epochs} Batch: {b+1}/{len(val_dataloader) + len(target_val_dataloader)}", end="\r")
                source_data = source_data.to(device)
                source_labels = source_labels.to(device)
                source_pred = model(source_data)
                loss_main = criterion(source_pred, source_labels)
                source_val_loss += loss_main.item()
                b += 1
            
            # Average the loss over all source validation batches
            source_val_loss /= len(val_dataloader)
            # Validation on Target Domain
            target_val_loss = 0.0
            for target_data in target_val_dataloader:
                print(f"Valid | Epoch: {epoch+1}/{num_epochs} Batch: {b+1}/{len(val_dataloader) + len(target_val_dataloader)}", end="\r")
                target_data = target_data.to(device)
                target_features = model.extract_features(target_data)
                domain_pred = domain_classifier(target_features, alpha=0)
                # Assuming you have a criterion for domain validation
                loss_domain = domain_criterion(domain_pred, torch.ones(target_data.size(0)).long().to(device))
                target_val_loss += loss_domain.item()
                b += 1

            # Average the loss over all target validation batches
            target_val_loss /= len(target_val_dataloader)

            if source_val_loss < best_loss:
                best_loss = source_val_loss
                torch.save(model.state_dict(), "adapted_model.pt")
                torch.save(domain_classifier.state_dict(), "domain_classifier.pt")
            # Update alpha and lambda_weight
            alpha, lambda_weight = update_hyperparameters(alpha, lambda_weight, source_val_loss, target_val_loss)
            wandb.log({"source_val_loss": source_val_loss,
            "target_val_loss": target_val_loss,
            "alpha": alpha,
            "lambda_weight": lambda_weight,
            "train_source_loss": train_source_loss,
            "train_target_loss": train_target_loss})
        # Print or log the validation losses
        print(f'Epoch [{epoch+1}/{num_epochs}], Source Val Loss: {source_val_loss:.4f}, Target Val Loss: {target_val_loss:.4f}')

def train_model():
    from train import Trainer

    model = TissuePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.125378, std=0.087051),
        ]
    )
    dataset = AngledAtlasDataset(
        Path(r"C:\Users\asoro\Desktop\angled_data"), transform=transforms
    )

    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    trainer = Trainer(
        model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda", project_name="tissue_resnet"
    )
    trainer.run(epochs=100)


if __name__ == "__main__":
   domain_adaptation()