import numpy as np
import cv2
import random
import os
from neuronal_network import NN

def load_data():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    set_path = r'C:\Users\Macie\PycharmProjects\NN-architecture-optimizer\set'

    arr = []
    images=[]
    labels=[]

    for photo in os.listdir(set_path):
        im = cv2.imread(set_path + '\\' + photo, 0)
        # reshaped = np.reshape(im, (50*50))
        images.append(im/255)
        labels.append(int('class1' in photo))
        # arr.append((int('class1' in photo), reshaped))

    trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, random_state=42)

    print(np.array(trainY).shape)
    print(np.array(trainX).shape)
    enc = OneHotEncoder(handle_unknown='ignore')
    trainYOHE=(enc.fit(np.array(trainY).reshape(-1,1))).transform(np.array(trainY).reshape(-1,1)).toarray()
    testYOHE=(enc.fit(np.array(testY).reshape(-1,1))).transform(np.array(testY).reshape(-1,1)).toarray()

    return trainX,trainYOHE,testX,testYOHE


trainX,trainY,testX,testY=load_data()

nn=NN([128,64,64,24])
nn.add_layer()
nn.train_network(np.array(trainX),trainY,epoch=20)
nn.evaluate_network(np.array(testX),testY)

print(f'acc: {nn.acc}\nloss: {nn.loss}')

# random.shuffle(arr)
# data = np.array(arr)
#
# train_set = data[:int(len(data)*3/4)]
# test_set = data[int(len(data)*3/4):]
#
# print(train_set.shape)
# print(test_set.shape)
