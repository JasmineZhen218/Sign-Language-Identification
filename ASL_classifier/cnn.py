import pickle
import numpy as np 
import pandas as pd
import cv2 as cv
from scipy import ndimage
import torch
import torch.nn.functional as F
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--file_path',help = "The location of the training and validation data.")
ap.add_argument('--lr',type = float, help = "learning rate.", default = 0.001)
ap.add_argument('--epochs',type = float, help = "number of epochs", default = 3000)
ap.add_argument('--batch_size',type = float, help = "batch size", default = 64)
args = ap.parse_args()


# load the data 
mapping = {
    'A': 0, 'B': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5,'G': 6, 'H': 7, 'I': 8,'K': 9, 'L': 10,'M': 11,'N': 12,'O': 13,'P': 14,'Q': 15,'R': 16,'S': 17, 'T': 18,'U': 19,'V': 20,'W': 21,'X': 22,'Y': 23,
     0:'A', 1:'B', 2:'C',3:'D', 4:'E', 5:'F',6:'G', 7:'H', 8:'I',9:'K', 10:'L',11:'M', 12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}
train = pd.read_csv(args.file_path+'Sign_MNIST/sign_mnist_train.csv')
test = pd.read_csv(args.file_path+'Sign_MNIST/sign_mnist_test.csv')
train.label = [i if i <= 8 else i-1 for i in train.label]
test.label = [i if i <= 8 else i-1 for i in test.label]
# extract features and labels
train_x = train.iloc[:,1:].to_numpy()
train_y = train.iloc[:,0].to_numpy()
dev_x = test.iloc[:,1:].to_numpy()
dev_y = test.iloc[:,0].to_numpy()
print("Training data:", train_x.shape, train_y.shape)
print("Validation data:", dev_x.shape, dev_y.shape)
# normalization
x_mean, x_std = np.mean(train_x, axis=0), np.std(train_x, axis=0)
train_x_n =  (train_x - x_mean)/x_std
dev_x_n = (dev_x -  x_mean)/x_std
# model
# Code for training models, or link to your Git repository
class BestModel(torch.nn.Module):
    def __init__(self, input_height, input_width, n_classes):
        super().__init__()
        self.input_height = input_height
        self.input_width= input_width
        self.n_classes = n_classes
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
        )    
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,self.n_classes)
        ) 
    def forward(self, x): 
        x = torch.reshape(x, (-1, self.input_height, self.input_width))
        x = torch.unsqueeze(x, 1)
        x = self.feature_extractor(x)
        x = torch.reshape(x, (x.shape[0], -1)) # flatten
        output = self.classifier(x)
        return output

def approx_train_acc_and_loss(model, train_data : np.ndarray, train_labels : np.ndarray) -> np.float64:
    idxs = np.random.choice(len(train_data), 4000, replace=False)
    x = torch.from_numpy(train_data[idxs].astype(np.float32))
    y = torch.from_numpy(train_labels[idxs].astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()

def dev_acc_and_loss(model, dev_data : np.ndarray, dev_labels : np.ndarray) -> np.float64:
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred.numpy()), loss.item()

def predict(model, dev_data : np.ndarray):
    x = torch.from_numpy(dev_data.astype(np.float32))
    logits = model(x)
    y_pred = torch.max(logits, 1)[1]
    return logits, y_pred

def accuracy(y : np.ndarray, y_hat : np.ndarray) -> np.float64:
    return np.sum(y == y_hat)/len(y)


# training
HEIGHT, WIDTH = 28, 28
LEARNING_RATE = args.lr #0.001
EPOCHS = args.epochs #3000
BATCH_SIZE = args.batch_size #64
model = BestModel(input_height = HEIGHT, input_width= WIDTH, n_classes= 24)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
Train_acc, Train_loss = [], []
Dev_acc, Dev_loss = [], []
for step in range(EPOCHS):
  i = np.random.choice(train_x.shape[0], size=BATCH_SIZE, replace=False)
  x = torch.from_numpy(train_x[i].astype(np.float32))
  y = torch.from_numpy(train_y[i].astype(np.int))        
  # Forward pass: Get logits for x
  logits = model(x)
  # Compute loss
  loss = F.cross_entropy(logits, y)
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()

  # log model performance every 100 epochs
  if step % 100 == 0:
    train_acc, train_loss = approx_train_acc_and_loss(model, train_x, train_y)
    dev_acc, dev_loss = dev_acc_and_loss(model, dev_x, dev_y)
    print("On step {}:\tTrain loss {:3f}\t|\tDev acc is {:.3f}".format(step, train_loss, dev_acc))
    Train_acc.append(train_acc)
    Train_loss.append(train_loss)
    Dev_acc.append(dev_acc)
    Dev_loss.append(dev_loss)

# Predict
train_acc, _ = dev_acc_and_loss(model, train_x, dev_y)
dev_acc, _ = dev_acc_and_loss(model, dev_x, dev_y)
print("CNN, Accuracy on training set is {:.2f}".format(train_acc))
print("CNN Forest, Accuracy on dev set is {:.2f}".format(dev_acc))