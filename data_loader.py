import numpy as np
import cv2
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from neuronal_network import NN
from constans_and_types import PROJECT_PATH


def load_data():
    set_path = PROJECT_PATH.joinpath('set')

    images=[]
    labels=[]

    for photo in os.listdir(set_path):
        im = cv2.imread(str(set_path.joinpath(photo)), 0)
        # reshaped = np.reshape(im, (50*50))
        images.append(im/255)
        labels.append(int('class1' in photo))
        # arr.append((int('class1' in photo), reshaped))

    trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, random_state=42)
    trainX, testX = np.array(trainX), np.array(testX)
    print(np.array(trainY).shape)
    print(np.array(trainX).shape)
    enc = OneHotEncoder(handle_unknown='ignore')

    trainYOHE=(enc.fit(np.array(trainY).reshape(-1,1))).transform(np.array(trainY).reshape(-1,1)).toarray()
    testYOHE=(enc.fit(np.array(testY).reshape(-1,1))).transform(np.array(testY).reshape(-1,1)).toarray()

    return trainX,trainYOHE,testX,testYOHE

if __name__ == "__main__":
    trainX,trainY,testX,testY=load_data()

    nn = NN([128, 64, 64, 24])
    nn.train_network(trainX, trainY, epoch=20)
    nn.evaluate_network(testX, testY)

    print(f'acc: {nn.acc}\nloss: {nn.loss}')

    # random.shuffle(arr)
    # data = np.array(arr)
    #
    # train_set = data[:int(len(data)*3/4)]
    # test_set = data[int(len(data)*3/4):]
    #
    # print(train_set.shape)
    # print(test_set.shape)
