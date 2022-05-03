import matplotlib.pyplot as plt 
import numpy as np
import torch
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils import tensorboard
from sklearn.model_selection import train_test_split
from torch import nn
import cv2
import os, pickle
from datetime import datetime

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial cnn w/ batch norm
        self.stageOneCNN = nn.Sequential(
            nn.Conv2d(1,64, (5,5), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,64, (5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,(5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64, (5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,(5,5), 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2, return_indices=True)
        )

        self.stageTwoCNN = nn.Sequential(
            nn.Conv2d(64, 128, (5,5), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, (5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, (5,5), 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2, return_indices=True)
        )

        self.stageThreeCNN = nn.Sequential(
            nn.Conv2d(128, 256, (5,5), 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5,5), 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2, return_indices=True)
        )

        self.stageFourCNN = nn.Sequential(
            nn.Conv2d(256, 512, (5,5), 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (5,5), 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (5,5), 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2, return_indices=True)
        )

        self.linearMap = nn.Sequential(
            nn.Linear(512 * 20 * 20, 2048),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        x, indicesOne = self.stageOneCNN(x)
        x, indicesTwo = self.stageTwoCNN(x)
        x, indicesThree = self.stageThreeCNN(x)
        x, indicesFour = self.stageFourCNN(x)
        x = x.reshape(-1, 512 * 20 * 20)
        x = self.linearMap(x)
        return x, [indicesOne, indicesTwo, indicesThree, indicesFour]

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearMap = nn.Sequential(
            nn.Linear(2048, 512 * 20 * 20),
            nn.LeakyReLU()
        )

        self.unpool = nn.MaxUnpool2d(2,2)

        self.stageFourDeconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (5,5), 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, (5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, (5,5), 1),
            nn.LeakyReLU(),
        )

        self.stageThreeDeconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (5,5), 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, (5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, (5,5), 1),
            nn.LeakyReLU()
        )

        self.stageTwoDeconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (5,5), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, (5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, (5,5), 1),
            nn.LeakyReLU()
        )

        self.stageOneDeconv = nn.Sequential(
            nn.ConvTranspose2d(64,64, (5,5), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,64, (5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,64,(5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,64, (5,5), 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,1,(5,5), 1),
            nn.LeakyReLU(),
        )

    def forward(self, x, indices, batch):
        x = self.linearMap(x)
        x = x.reshape(batch, 512, 20, 20)
        x = self.unpool(x, indices[3])
        x = self.stageFourDeconv(x)
        x = self.unpool(x, indices[2])
        x = self.stageThreeDeconv(x)
        x = torch.nn.functional.pad(input=x, pad=(0,1,1,0), mode='constant', value=0.0)
        x = self.unpool(x, indices[1])
        x = self.stageTwoDeconv(x)
        x = self.unpool(x, indices[0])
        x = self.stageOneDeconv(x)
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

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# Setup params
paramsToOptimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# summary(model, input_size=(1, 512, 512))\
# Optimizer and Loss
optimizer = torch.optim.SGD(paramsToOptimize, lr=0.001, momentum=0.9)
loss_fn = torch.nn.MSELoss()

def trainEpoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainingLoader):
        # Every data instance is an input + label pair
        inputs = data.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        encoded, indices = encoder(inputs)
        decoded = decoder(encoded, indices, len(inputs))
        # Compute the loss and its gradients
        loss = loss_fn(decoded, inputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(trainingLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == '__main__':
    # Transformations on images
    transforms = transforms.Compose([transforms.ToTensor()])
    # Read the png locations
    nrrdPath = "C:/Users/Alec/.belljar/nrrd/png_half/"
    fileList = os.listdir(nrrdPath) # path to flat pngs
    absolutePaths = [nrrdPath + p for p in fileList]
    # Load all the images into memory
    allSlices = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths[:int(len(absolutePaths)*0.05)]] #[:int(len(absolutePaths)*0.05)]
    # Split this up into t and v
    trainingImages, validationImages = train_test_split(allSlices, test_size=0.2)
    trainingDataset, validationDataset = Nissl(trainingImages, transform=transforms), Nissl(validationImages, transform=transforms)
    # Now construct data loaders for batch training
    trainingLoader, validationLoader = DataLoader(trainingDataset, batch_size=2, shuffle=True), DataLoader(validationDataset, batch_size=2, shuffle=True)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = tensorboard.SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        encoder.train(True)
        decoder.train(True)
        avg_loss = trainEpoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        encoder.train(False)
        decoder.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validationLoader):
            vinputs = vdata.to(device)
            encoded, indices = encoder(vinputs)
            decoded = decoder(encoded, indices, len(vinputs))
            vloss = loss_fn(decoded, vinputs)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(encoder.state_dict(), model_path)
            torch.save(decoder.state_dict(), model_path)

        epoch_number += 1


def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = validationDataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = validationDataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()   