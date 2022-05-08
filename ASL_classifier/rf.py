from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np 
import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--file_path',help = "The location of the training and validation data.")
args = ap.parse_args()

# load the data 
mapping = {
    'A': 0, 'B': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5,'G': 6, 'H': 7, 'I': 8,'K': 9, 'L': 10,'M': 11,'N': 12,'O': 13,'P': 14,'Q': 15,'R': 16,'S': 17, 'T': 18,'U': 19,'V': 20,'W': 21,'X': 22,'Y': 23,
     0:'A', 1:'B', 2:'C',3:'D', 4:'E', 5:'F',6:'G', 7:'H', 8:'I',9:'K', 10:'L',11:'M', 12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}
train = pd.read_csv(dir+'Sign_MNIST/sign_mnist_train.csv')
test = pd.read_csv(dir+'Sign_MNIST/sign_mnist_test.csv')
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
# fit
clf_rf = RandomForestClassifier( random_state=0)
clf_rf.fit(train_x_n, train_y)
# predict
train_y_hat = clf_rf.predict(train_x_n)
dev_y_hat = clf_rf.predict(dev_x_n)
print("Random Forest, Accuracy on training set is {:.2f}".format(np.mean(train_y_hat == train_y)))
print("Random Forest, Accuracy on dev set is {:.2f}".format(np.mean(dev_y_hat == dev_y)))
