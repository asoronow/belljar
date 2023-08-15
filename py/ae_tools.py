import torch
from torchvision import transforms
from torch import nn
import cv2
import numpy as np
import os, pickle
from scipy import spatial
from demons import match_histograms
import nrrd
import SimpleITK as sitk
from pathlib import Path
from model import TissueAutoencoder


class Nissl:
    def __init__(self, images, transform=None, labels=None):
        self.images = images
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def getPath(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            img = self.transform(img)
        return img


def make_predictions(
    dapiImages, dapiLabels, modelPath, embeddPath, nrrdPath, hemisphere=True
):
    """Use the encoded sections and atlas embeddings to register brain regions"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder model
    encoder = TissueAutoencoder()
    checkpoint = torch.load(modelPath)
    encoder.load_state_dict(checkpoint)
    encoder.to(device)
    encoder.eval()

    # Load the atlas embeddings
    with open(embeddPath, "rb") as f:
        embeddings = pickle.load(f)

        # Normalize the dapi images to atlas range
        atlas, atlasHeader = nrrd.read(
            str(nrrdPath / f"ara_nissl_10_all.nrrd"), index_order="C"
        )
        sample = atlas[800, :, :]
        sample = sitk.GetImageFromArray(sample)
        sample = sitk.Cast(sample, sitk.sitkUInt8)

        matched = []
        for image in dapiImages:
            dapi = sitk.GetImageFromArray(image)
            matched_dapi = match_histograms(dapi, sample)
            matched_dapi = sitk.GetArrayFromImage(matched_dapi)
            matched.append(matched_dapi)

        t = transforms.Compose([transforms.ToTensor()])
        dataset = Nissl(matched, transform=t, labels=dapiLabels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        most_accurate_predictions = {}

        for i, sample in enumerate(loader):
            sample = sample.to(device)
            with torch.no_grad():
                latent_representation = encoder.encode_to_latent(sample)
                latent_representation = (
                    latent_representation.cpu()
                    .numpy()
                    .reshape(
                        latent_representation.shape[1],
                    )
                )

            # Find the atlas embedding with the smallest cosine distance to the image's latent representation
            max_similarity = float("-inf")
            best_match = None
            for name, embedding in embeddings.items():
                similarity = 1 - spatial.distance.cosine(
                    latent_representation, embedding
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name

            # Store the best match for this image
            most_accurate_predictions[dataset.getPath(i)] = int(
                best_match.split(".")[0].split("_")[1]
            )

        return most_accurate_predictions


def create_png_dataset():
    """
    Make a dataset of PNGs from the NRRD files to utilize in training
    """

    def rotate_image(img):
        """Rotate a random amount between -10 and 10 degrees"""
        angle = np.random.randint(-10, 10)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    dataset_path = Path(r"C:\Users\Alec\.belljar\dataset")
    # Load the atlas
    atlas, _ = nrrd.read(
        Path(r"C:\Users\Alec\.belljar\nrrd\ara_nissl_10_all.nrrd"), index_order="C"
    )
    z, y, x = atlas.shape

    right_atlas = atlas[:, :, ::-1]
    left_atlas = atlas[:, :, :]

    while len(os.listdir(dataset_path)) < 5000:
        for i in range(50, z - 50, 1):
            # Pad 100 pixels on each side
            right = right_atlas[i, :, :]
            left = left_atlas[i, :, :]
            right = np.pad(right, 25, "constant", constant_values=0)
            left = np.pad(left, 25, "constant", constant_values=0)

            # Roate each image a random amount
            right = rotate_image(right)
            left = rotate_image(left)

            # Resize the images to 256x256
            right = cv2.resize(right, (256, 256))
            left = cv2.resize(left, (256, 256))

            # Random string to make sure that the images are not overwritten
            rand = np.random.randint(0, 1000000)
            rand2 = np.random.randint(0, 1000000)
            # Save the images
            cv2.imwrite(str(dataset_path / f"right_{rand}.png"), right)
            cv2.imwrite(str(dataset_path / f"left_{rand2}.png"), left)

    # Then just add the straight on images
    for i in range(50, z - 50, 1):
        # Pad 100 pixels on each side
        right = right_atlas[i, :, :]
        left = left_atlas[i, :, :]
        right = np.pad(right, 25, "constant", constant_values=0)
        left = np.pad(left, 25, "constant", constant_values=0)

        # Resize the images to 256x256
        right = cv2.resize(right, (256, 256))
        left = cv2.resize(left, (256, 256))

        # Random string to make sure that the images are not overwritten
        rand = np.random.randint(0, 1000000)
        rand2 = np.random.randint(0, 1000000)
        # Save the images
        cv2.imwrite(str(dataset_path / f"right_{rand})_normal.png"), right)
        cv2.imwrite(str(dataset_path / f"left_{rand2}_normal.png"), left)


def clean_junk(dataset_path):
    """Removes all images where there is less than 10% of the image that is not zero"""
    files = os.listdir(dataset_path)
    for file in files:
        if file.endswith(".png"):
            img = cv2.imread(str(dataset_path / file), cv2.IMREAD_GRAYSCALE)
            if np.count_nonzero(img) < 0.1 * img.size:
                os.remove(str(dataset_path / file))


def make_embeddings(dataset_path):
    embeddings = {}

    model = TissueAutoencoder()
    checkpoint = torch.load(Path(r"C:\Users\Alec\Projects\belljar\best_model.pt"))
    model.load_state_dict(checkpoint)
    model.to("cuda:0")
    model.eval()

    with torch.no_grad():
        for file in os.listdir(dataset_path):
            if file.endswith(".png"):
                img = cv2.imread(str(dataset_path / file), cv2.IMREAD_GRAYSCALE)

                # Normalize the image to [0, 1]
                img = img / 255.0

                # Convert to tensor, add batch and channel dimensions, and send to device
                img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
                img_tensor = img_tensor.to("cuda:0")

                out = model.encode_to_latent(img_tensor)
                out = out.cpu().numpy().squeeze()  # Remove singleton dimensions
                embeddings[file] = out

    with open(Path(r"C:\Users\Alec\.belljar\embeddings\embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings for {len(embeddings)} images.")


if __name__ == "__main__":
    # create_png_dataset()
    make_embeddings(Path(r"C:\Users\Alec\.belljar\solo-dataset"))
