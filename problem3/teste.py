import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score as BACC

class ImageDataset(Dataset):
    def __init__(self):
        xtrain = np.load("Xtrain_Classification_Part1.npy")
        ytrain = np.load("Ytrain_Classification_Part1.npy")
        xtest = np.load("Xtest_Classification_Part1.npy")


        xtrain_len = len(xtrain)
        ytrain_len = len(ytrain)
        xtest_len = len(xtest)

        #Reshape Images
        self.xtrain = xtrain.reshape((xtrain_len,50,50))
        self.xtest = xtest.reshape((xtest_len,50,50))

        self.ytrain = ytrain.reshape(ytrain_len).astype(np.int8)

    def __len__(self):
        return len(self.xtrain)

    def __getitem__(self, idx):
        image = self.xtrain[idx, :, :]
        label = self.ytrain[idx]

        return image, label



train_set = ImageDataset()

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)



for data in train_loader:
    image, label = data

    print(f"Image size: {image.shape}, label: {label}")
