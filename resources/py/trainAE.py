from msilib import sequence
from random import randint
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os, pickle
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8 * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * 2, 16 * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(16 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16 * 2 , 32 * 2, 3, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(30752 * 2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 30752 * 2 ),
            nn.LeakyReLU(inplace=True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32*2, 31 , 31))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 16 * 2, 3, 
            stride=2, output_padding=1),
            nn.BatchNorm2d(16 * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16 * 2, 8 * 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8 * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(8 * 2, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
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

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

    ### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder, test_dataset, device,n=4):
    plt.figure(figsize=(5,10))
    for i in range(n):
      ax = plt.subplot(n,i+1,2)
      img = test_dataset[randint(0,len(test_dataset))].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(n, i + 1 + n, 2)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

def runTraining(num_epochs=300):
    fileList = os.listdir("../nrrd/png_half") # path to flat pngs
    absolutePaths = [os.path.join('../nrrd/png_half', p) for p in fileList]
    
    allSlices = [cv2.cvtColor(cv2.resize(cv2.imread(p), (256,256)), cv2.COLOR_BGR2GRAY) for p in absolutePaths] #[:int(len(absolutePaths)*0.05)]
   
    train_dataset, test_dataset = train_test_split(allSlices, test_size=0.2)
    train_dataset, test_dataset = Nissl(train_dataset, transform=transforms.ToTensor()), Nissl(test_dataset, transform=transforms.ToTensor())

    m=len(train_dataset)

    train_data, val_data = random_split(train_dataset, [m-int(m*0.2), int(m*0.2)])
    batch_size=256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, pin_memory=True)

    loss_fn = torch.nn.MSELoss()
    lr= 0.0001
    d = 16

    encoder = Encoder(encoded_space_dim=d)
    decoder = Decoder(encoded_space_dim=d)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder,decoder,device, train_loader,loss_fn,optim)
        val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)

    torch.save(encoder.state_dict(), 'encoder_half.pt')
    torch.save(decoder.state_dict(), 'decoder_half.pt')

    plt.figure(figsize=(10,8))
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.semilogy(diz_loss['train_loss'], label='Train')
    plt.semilogy(diz_loss['val_loss'], label='Valid')
    plt.show()


def loadModels(d=16):
    encoder = Encoder(encoded_space_dim=d)
    decoder = Decoder(encoded_space_dim=d)
    
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)
    
    encoder.load_state_dict(torch.load('encoder_half.pt'))
    encoder.eval()

    decoder.load_state_dict(torch.load('decoder_half.pt'))
    decoder.eval()

    return encoder, decoder, device

def embedAtlasDataset():
    fileList = os.listdir("../nrrd/png_half") # path to flat pngs
    absolutePaths = [os.path.join('../nrrd/png_half', p) for p in fileList]
    
    allSlices = [cv2.cvtColor(cv2.resize(cv2.imread(p), (256,256)), cv2.COLOR_BGR2GRAY) for p in absolutePaths] #[:int(len(absolutePaths)*0.05)]
    
    encoder, decoder, device = loadModels()
    
    atlasDataset = Nissl(allSlices, labels=fileList, transform=transforms.ToTensor())
    atlasLoader = torch.utils.data.DataLoader(atlasDataset, batch_size=1, pin_memory=True)
    atlasEncoding = []
    for batch in atlasLoader:
        batch = batch.to(device)
        encoded = encoder(batch)
        atlasEncoding.append(encoded.detach().cpu().squeeze().numpy())
    
    finalEncoding = {}
    for idx, vector in enumerate(atlasEncoding):
        finalEncoding[absolutePaths[idx]] = vector

    return finalEncoding

def compareSampleImages(images, half=False):
    encoder, decoder, device = loadModels()

    atlasEncoding = embedAtlasDataset()
    # with open("wholebrain_embedings_paths.pkl", "rb") as f:
    #     atlasEncoding = pickle.load(f)

    sampleDataset = Nissl(images, transform=transforms.ToTensor())
    sampleLoader = torch.utils.data.DataLoader(sampleDataset, batch_size=1, pin_memory=True)
    sampleEncoding = []
    for batch in sampleLoader:
        batch = batch.to(device)
        encoded = encoder(batch)
        sampleEncoding.append(encoded.detach().cpu().squeeze().numpy())

    knn = NearestNeighbors(metric="euclidean")
    knn.fit(list(atlasEncoding.values()))
    _, indicies = knn.kneighbors(sampleEncoding)
    paths = list(atlasEncoding.keys())
    for idx in indicies.flatten():
        print(paths[idx])
        cv2.imshow('Match', cv2.imread(paths[idx]))
        cv2.waitKey(0)

if __name__ == '__main__':
    compareSampleImages([cv2.cvtColor(cv2.resize(cv2.imread('M466_s037.png'), (256,256)), cv2.COLOR_BGR2GRAY)])
    #runTraining()