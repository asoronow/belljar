import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import cv2
import os, pickle
from scipy import spatial
from demons import match_histograms
import nrrd
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import rotate
import time

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial cnn w/ batch norm
        self.stageOneCNN = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.LeakyReLU(),
        )

        self.stageTwoCNN = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.LeakyReLU(),
        )

        self.stageThreeCNN = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, (3, 3)),
            nn.LeakyReLU(),
        )

        self.stageFourCNN = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.linearMap = nn.Sequential(nn.Linear(256 * 28 * 28, 2048), nn.LeakyReLU())

    def forward(self, x):
        x = self.stageOneCNN(x)
        x = self.stageTwoCNN(x)
        x = self.stageThreeCNN(x)
        x = self.stageFourCNN(x)
        x = self.flatten(x)
        x = self.linearMap(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearMap = nn.Sequential(nn.Linear(2048, 256 * 28 * 28), nn.LeakyReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 28, 28))

        self.stageFourDeconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (3, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, (3, 3), 2),
            nn.LeakyReLU(),
        )

        self.stageThreeDeconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, (3, 3), 2, 1),
            nn.LeakyReLU(),
        )

        self.stageTwoDeconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3)), nn.BatchNorm2d(64), nn.LeakyReLU()
        )

        self.stageTwoOutput = nn.ConvTranspose2d(64, 32, (3, 3), 2, 1)

        self.stageOneDeconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (3, 3)), nn.BatchNorm2d(32), nn.LeakyReLU()
        )

        self.stageOneOutput = nn.ConvTranspose2d(32, 1, (3, 3), 2, 1)

    def forward(self, x):
        x = self.linearMap(x)
        x = self.unflatten(x)
        x = self.stageFourDeconv(x)
        x = self.stageThreeDeconv(x)
        x = self.stageTwoDeconv(x)
        x = self.stageTwoOutput(x, output_size=(254, 254))
        x = torch.nn.functional.leaky_relu(x)
        x = self.stageOneDeconv(x)
        x = self.stageOneOutput(x, output_size=(512, 512))
        x = torch.nn.functional.leaky_relu(x)
        return x


class Nissl(Dataset):
    def __init__(self, images, labels=None, transform=None, target_transform=None):
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

    def getPath(self, idx):
        label = self.labels[idx]
        return label


def trainEpoch(
    epoch_index, tb_writer, trainingLoader, optimizer, device, encoder, decoder, loss_fn
):
    running_loss = 0.0
    last_loss = 0.0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainingLoader):
        # Every data instance is an input + label pair
        inputs = data.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        encoded = encoder(inputs)
        decoded = decoder(encoded)
        # Compute the loss and its gradients
        loss = loss_fn(decoded, inputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 50  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(trainingLoader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def plot_ae_outputs(encoder, decoder, images, n=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(16, 4.5))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        validationDataset = Nissl(images, transform=t)
        img = validationDataset[i].to(device)
        img = img[None, :]
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            out = encoder(img)
            rec_img = decoder(out)
        plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original images")
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed images")
    plt.show()


def make_predictions(dapiImages, dapiLabels, modelPath, embeddPath, nrrdPath, hemisphere=True):
    """Use the encoded sections and atlas embeddings to register brain regions"""
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    encoder = nn.DataParallel(Encoder())
    encoder.load_state_dict(torch.load(modelPath, map_location=device))
    encoder.eval()
    encoder.to(device)
    # load the atlas embeddings
    embeddings = {}
    with open(embeddPath, "rb") as f:
        embeddings = pickle.load(f)
        for name, e in embeddings.items():
            embeddings[name] = e

    # Normalize the dapi images to atlas range
    atlas, atlasHeader = nrrd.read(str(nrrdPath / f"ara_nissl_10_all.nrrd"), index_order="C")
    x, y, z = atlas.shape

    sample = atlas[800, : , :]
    # convert atlas to sitk
    sample = sitk.GetImageFromArray(sample)
    # convert data type
    sample = sitk.Cast(sample, sitk.sitkUInt8)
    matched = []
    for i in range(len(dapiImages)):
        # convert to sitk
        dapi = sitk.GetImageFromArray(dapiImages[i])
        matched_dapi = match_histograms(dapi, sample)
        # convert back to numpy from sitk
        matched_dapi = sitk.GetArrayFromImage(matched_dapi)
        matched.append(matched_dapi)
  
    # Load the images
    t = transforms.Compose([transforms.ToTensor()])
    dataset = Nissl(matched, transform=t, labels=dapiLabels)
    # Create a dataloader
    # We use a batch size of 1 to make sure that the images are loaded in memory
    # This is necessary for the dataloader to work properly on single GPUs
    similarity = {}
    for i in range(len(dataset)):
        img = dataset[i].to(device)
        img = img[None, :]
        with torch.no_grad():
            out = encoder(img).cpu().numpy()
            out = out.reshape(
                out.shape[1],
            )

            similarity[dataset.getPath(i)] = {}
            for name, e in embeddings.items():
                e = e.reshape(
                    e.shape[1],
                )

                similarity[dataset.getPath(i)][name] = spatial.distance.euclidean(
                    out, e
                )

    # Select the best sections along that angle
    best = {}
    idealAngle = 0
    for name, scores in similarity.items():
        ordered = sorted(scores, key=scores.get)
        section = None
        for result in ordered[:5]:
            v = result.split("_")
            s = int(v[3].split(".")[0])
            if int(v[2]) == idealAngle:
                section = result
                break

        if section == None:
            sectionEmbedding = embeddings[ordered[0]]
            sectionEmbedding = sectionEmbedding.reshape(
                sectionEmbedding.shape[1],
            )
            matches = {}
            for atlasName, e in embeddings.items():
                v = atlasName.split("_")
                if int(v[2]) == idealAngle:
                    e = e.reshape(
                        e.shape[1],
                    )
                    matches[atlasName] = spatial.distance.euclidean(sectionEmbedding, e)
            best[name] = min(matches, key=matches.get)
        else:
            best[name] = section

    # Prep for alignment by converting to integer slices
    for sectionName, matchName in best.items():
        best[sectionName] = int(matchName.split("_")[3].split(".")[0])

    return best, idealAngle



def createEmbeddings(pngFolder, embeddingFileName, modelPath):
    """
    Create a pkl with embeddings from a folder of section slices
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = nn.DataParallel(Encoder())
    encoder.load_state_dict(torch.load(modelPath, map_location=device))
    encoder.eval()
    encoder.to(device)
    files = os.listdir(pngFolder)
    absolutePaths = [pngFolder + p for p in files]
    allSlices = [
        cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths
    ]  # [:int(len(absolutePaths)*0.05)]
    allLabels = [p for p in files]
    t = transforms.Compose([transforms.ToTensor()])
    dataset = Nissl(allSlices, transform=t, labels=allLabels)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    embeddings = {}
    for i, data in enumerate(loader):
        inputs = data.to(device)
        encoded = encoder(inputs)
        embeddings[dataset.getPath(i)] = encoded.cpu().detach().numpy()

    with open(embeddingFileName, "wb") as f:
        pickle.dump(embeddings, f)

def rotate_atlas(atlas, angle_deg):
    '''
    Uses scipy to quickly rotate the atlas volume to simulate a cut at a specific angle
    
    Parameters:
        atlas (np.ndarray): the atlas volume
        angle_deg (float): the angle to cut the atlas at
    Returns:
        np.ndarray: the rotated atlas volume
    '''
    return rotate(atlas, angle_deg, axes=(0, 1), order=0, reshape=False)
