# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
#import plotly.express as px
import matplotlib.pylab as plt
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

from PIL import Image
import cv2
import os
import pickle
import seaborn as sns



# this is working directory
def dumpclassifier(filename, model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)

def rand_img(dir):
    IMG_DIR = dir+"/images"
    img_array_total = np.empty([2046, 784])
    i = 0
    for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR, img), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            continue
        img_pil = Image.fromarray(img_array)
        img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))

        img_array = (img_28x28.flatten())

        img_array = img_array.reshape(-1, 1).T
        img_array_total[i] = img_array
        i = i + 1

    zero_ish = np.zeros((2046,), dtype=int)
    img_array_total = np.concatenate((zero_ish[:, np.newaxis], img_array_total), axis=1)
    arr = ['label']
    for i in range(784):
        arr.append('pixel' + str(i))
    test_rnd_image = pd.DataFrame(img_array_total, columns=arr)
    test_rnd_image = test_rnd_image.astype(int)

    return test_rnd_image


def load_data(dir, test_rnd_image):
    train = pd.read_csv(dir + "/sign_mnist_train.csv")

    test = pd.read_csv(dir + "/sign_mnist_train.csv")
    train_digit = pd.read_csv(dir + "/train.csv")

    test_digit = pd.read_csv(dir +"/test.csv")
    train_fashion = pd.read_csv(dir + "/fashion-mnist_train.csv")
    test_fashion = pd.read_csv(dir + "/fashion-mnist_test.csv")

    # augment the x_test by 1 dim
    test['label'] = 1
    test_digit['label'] = 0
    test_fashion['label'] = 0
    train['label'] = 1
    train_digit['label'] = 0
    train_fashion['label'] = 0

    train = train[:2046]

    train = pd.concat([train, test_rnd_image])
    test = pd.concat([test, test_digit, test_fashion])

    train = train.fillna(0)
    test = test.fillna(0)

    train_label = np.zeros([29500, 1])
    test_label = np.zeros([65455, 1])

    test = test.sample(frac=1)
    train = train.sample(frac=1)

    for i in range(2046):
        test_label[i] = 1
    test = test.iloc[:, :-1]
    train = train.iloc[:, :-1]

    return test,train,test_label,train_label

def split_data(test_x, test_y,train_x):

    dev_x, blind_x, dev_y, blind_y = train_test_split(test_x, test_y, test_size=.8, stratify=test_y)

    train_x = train_x.to_numpy()
    dev_x = dev_x.to_numpy()
    blind_x = blind_x.to_numpy()
    train_x = train_x.reshape(-1, 28, 28, 1)
    dev_x = dev_x.reshape(-1, 28, 28, 1)
    blind_x = blind_x.reshape(-1, 28, 28, 1)


    return dev_x, dev_y, blind_x, blind_y, train_x


def ResNet50(labels):
    classes = len(labels)
    input_shape = (28, 28, 1)
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1))(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='sigmoid')(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


def train_procedure(train_x, train_y, dev_x, dev_y, blind_x, blind_y,labels):

    model = ResNet50(labels)

    train_y = pd.pandas.get_dummies(train_y)
    dev_y = pd.pandas.get_dummies(dev_y)
    blind_y = pd.pandas.get_dummies(blind_y)

    model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

    best_model = tf.keras.callbacks.ModelCheckpoint("best.h5", monitor="val_accuracy")
    history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=64),
                                  validation_data=datagen.flow(dev_x, dev_y), epochs=10, callbacks=[best_model])
    model.load_weights("best.h5")

    dumpclassifier('signDetector.pkl', model)

    return history

def plot_loss(history):
    X = range(len(history.history["loss"]))
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    plt.style.use("seaborn")
    plt.figure()
    plt.title("Loss history")
    sns.lineplot(x=X, y=train_loss, label="Train", marker="o")
    sns.lineplot(x=X, y=val_loss, label="val", marker="o")
    plt.show()
    plt.figure()
    plt.title("Accuracy history")
    sns.lineplot(x=X, y=train_acc, label="Train", marker="o")
    sns.lineplot(x=X, y=val_accuracy, label="val", marker="o")
    plt.show()

def main():
    # fill in your directory here
    dir = "/content/drive/MyDrive/ML_final/Sign_MNIST"
    test_rnd_image = rand_img(dir)
    test, train, test_label, train_label = load_data(dir,test_rnd_image)
    train_x = train.drop("label", axis=1)
    train_y = train["label"]
    test_x = test.drop("label", axis=1)
    test_y = test["label"]

    labels = train["label"].value_counts().sort_index(ascending=True)


    dev_x, dev_y, blind_x, blind_y, train_x = split_data(test_x,test_y,train_x)
    history = train_procedure(train_x, train_y, dev_x, dev_y, blind_x, blind_y,labels)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
