from random import randint
import matplotlib.pyplot as plt 
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from torch import nn
import cv2
import os, pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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

def plot_ae_outputs(encoder, decoder, test_dataset, device,n=4):
    fig, ax = plt.subplots(4, 2, sharex='col', sharey='row',figsize=(5, 10))
    ax[0, 0].title.set_text('Real')
    ax[0, 1].title.set_text('Prediksi')
    for i in range(n):    
      img = test_dataset[randint(0,len(test_dataset))].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      ax[i, 0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax[i, 0].get_xaxis().set_visible(False)
      ax[i, 0].get_yaxis().set_visible(False)  
      ax[i, 1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax[i, 1].get_xaxis().set_visible(False)
      ax[i, 1].get_yaxis().set_visible(False)  

    plt.show()

def runTraining(num_epochs=600):
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
    d = 2056

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

    torch.save(encoder.state_dict(), '../models/encoder_half.pt')
    torch.save(decoder.state_dict(), '../models/decoder_half.pt')

    plt.figure(figsize=(10,8))
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.semilogy(diz_loss['train_loss'], label='Train')
    plt.semilogy(diz_loss['val_loss'], label='Valid')
    plt.show()


def loadModels(d=2056):
    encoder = Encoder(encoded_space_dim=d)
    decoder = Decoder(encoded_space_dim=d)
    
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)
    
    encoder.load_state_dict(torch.load('../models/encoder_half.pt'))
    encoder.eval()

    decoder.load_state_dict(torch.load('../models/decoder_half.pt'))
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

    atlasEncoding = {}
    with open("half_embedings.pkl", "rb") as f:
        atlasEncoding = pickle.load(f)

    sampleDataset = Nissl(images, transform=transforms.ToTensor())
    sampleLoader = torch.utils.data.DataLoader(sampleDataset, batch_size=1, pin_memory=True)
    sampleEncoding = []
    for batch in sampleLoader:
        # cv2.imshow("image", batch.cpu().squeeze().numpy())
        batch = batch.to(device)
        encoded = encoder(batch)
        sampleEncoding.append(encoded.detach().cpu().squeeze().numpy())

    scores = {}
    for path, embedding in atlasEncoding.items():
        scores[path] = cosine_similarity([embedding], [sampleEncoding[0]])
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    for path, score in scores.items():
        cv2.imshow("Match", cv2.imread(path))
        cv2.waitKey(0)
if __name__ == '__main__':
    compareSampleImages([cv2.cvtColor(cv2.resize(cv2.imread('sample_dapi.png'), (256,256)), cv2.COLOR_BGR2GRAY)])
    # runTraining()
    # encoder, decoder, device = loadModels()
    # fileList = os.listdir("../nrrd/png_half") # path to flat pngs
    # absolutePaths = [os.path.join('../nrrd/png_half', p) for p in fileList]
    # allSlices = [cv2.cvtColor(cv2.resize(cv2.imread(p), (256,256)), cv2.COLOR_BGR2GRAY) for p in absolutePaths[:int(len(absolutePaths)*0.05)]] #[:int(len(absolutePaths)*0.05)]
    # atlasDataset = Nissl(allSlices, labels=fileList, transform=transforms.ToTensor())
    # plot_ae_outputs(encoder, decoder, atlasDataset, device)
    # with open("half_embedings.pkl", 'wb') as file:
    #     embeddings = embedAtlasDataset()
    #     pickle.dump(embeddings, file)