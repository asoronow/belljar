import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from model import Autoencoder
import h5py
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.spatial.distance import directed_hausdorff
from PIL import Image


def save_matches_to_image(matches, filename, image_size):
    # Calculate the size of the output image
    num_matches = len(matches)
    output_size = (image_size[1] * num_matches, image_size[0] * 2)

    # Create a blank grayscale image with the calculated size
    blank_image = np.zeros(output_size, dtype=np.uint8)

    # Loop over the matches and extract the original and match images
    for i, match in enumerate(matches):
        og_image = Image.open(match["image"]).convert("L")
        match_image = Image.open(match["match"]).convert("L")
        og_image = og_image.resize(image_size, resample=Image.BILINEAR)
        match_image = match_image.resize(image_size, resample=Image.BILINEAR)

        # Paste the original and match images side by side into the blank image
        y = i * image_size[1]
        blank_image[y : y + image_size[1], : image_size[0]] = np.asarray(og_image)
        blank_image[y : y + image_size[1], image_size[0] :] = np.asarray(match_image)

    # Save the image to file
    Image.fromarray(blank_image).save(filename)


def createEmbeddings(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = Autoencoder()
    model.to(device)
    model.load_state_dict(torch.load("autoencoder.pth"))
    # Load the images
    dataset = ImageFolder(
        root=images,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    model.eval()
    with torch.no_grad():
        batch = 0
        for inputs, _ in dataloader:
            embeddings = []
            inputs = inputs.to(device)
            outputs = model.encoder(inputs)

            def get_angle(path):
                return path.split("\\")[-1].split("_")[2]

            def get_class(path):
                return path.split("\\")[-2]

            for i in range(outputs.shape[0]):
                j = i + batch * 64
                out = {
                    "tensor": outputs[i].cpu().numpy(),
                    "angle": get_angle(dataset.samples[j][0]),
                    "slice": dataset.samples[j][0]
                    .split("\\")[-1]
                    .split("_")[3]
                    .split(".")[0],
                    "class": get_class(dataset.samples[j][0]),
                }
                embeddings.append(out)
            # append to hdf5 file, use compression
            with h5py.File("embeddings.hdf5", "a") as f:
                for embedding in embeddings:
                    if embedding["class"] not in f:
                        f.create_group(embedding["class"])
                    if embedding["angle"] not in f[embedding["class"]]:
                        f[embedding["class"]].create_group(embedding["angle"])
                    f[embedding["class"]][embedding["angle"]].create_dataset(
                        embedding["slice"],
                        data=embedding["tensor"],
                        dtype="f",
                        compression="gzip",
                        compression_opts=4,
                    )
            batch += 1


def progressBar(current, total):
    percent = current / total
    bar = "#" * int(percent * 20)
    print(f"[{bar:<20}] {percent * 100:.2f}%", end="\r")


if __name__ == "__main__":
    # createEmbeddings("C:\\Users\\Alec\\Projects\\aba-nrrd\\data")
    # read embeddings from hdf5 file

    with h5py.File("embeddings.hdf5", "r") as f:
        images = "C:\\Users\\Alec\\Projects\\belljar-testing\\r151"
        dataset = ImageFolder(
            root=images,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ]
            ),
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        model = Autoencoder()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.load_state_dict(torch.load("autoencoder.pth"))
        model.eval()
        section = "hemisphere"
        matches = []
        prior_slice = 0
        with torch.no_grad():
            c = 0
            j = 0

            # find angle first
            consensus = {}
            print("Finding angle...")
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model.encoder(inputs)

                for angle in f[section]:
                    previous = 0
                    for sliceNumber in f[section][angle]:
                        if int(sliceNumber) < previous + 50:
                            continue
                        embedding = torch.from_numpy(
                            f[section][angle][sliceNumber][:]
                        ).to(device)
                        dist = (
                            1
                            - torch.nn.functional.cosine_similarity(
                                outputs.view(256, 32**2), embedding.view(256, 32**2)
                            )
                            .cpu()
                            .numpy()
                        )
                        dist = np.mean(dist)
                        if angle not in consensus:
                            consensus[angle] = []
                        consensus[angle].append(dist)
                        previous = int(sliceNumber)
            consensus = {k: np.mean(v) for k, v in consensus.items()}
            # get the best angle (aka minimum distance)
            consensus = min(consensus, key=consensus.get)
            total = len(f[section][consensus]) * len(dataset.samples)
            print(f"Best angle: {consensus}")
            print("Finding best matches...")
            for inputs, _ in dataloader:
                bestMatch = None
                bestDist = np.inf
                inputs = inputs.to(device)
                outputs = model.encoder(inputs)
                for sliceNumber in f[section][consensus]:
                    progressBar(j, total)
                    if int(sliceNumber) < prior_slice + 1:
                        j += 1
                        continue

                    embedding = torch.from_numpy(
                        f[section][consensus][sliceNumber][:]
                    ).to(device)
                    dist = (
                        1
                        - torch.nn.functional.cosine_similarity(
                            outputs.view(256, 32**2),
                            embedding.view(256, 32**2),
                        )
                        .cpu()
                        .numpy()
                    )
                    dist = np.sum(dist)
                    if dist < bestDist:
                        bestMatch = f"{section}/{consensus}/{int(sliceNumber)}"
                        bestDist = dist
                    j += 1

                print(f"Best match: {bestMatch} ({bestDist})")
                prior_slice = int(bestMatch.split("/")[2])
                matches.append(
                    {
                        "distance": bestDist,
                        "sliceNumber": int(bestMatch.split("/")[2]),
                        "image": dataset.samples[c][0],
                        "match": f"c:\\Users\\Alec\\Projects\\aba-nrrd\\data\\{section}\\r_nissil_{bestMatch.split('/')[1]}_{bestMatch.split('/')[2]}.png",
                    }
                )
                c += 1
        save_matches_to_image(matches, "results.png", (512, 512))
