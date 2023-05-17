import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

from constans_and_types import PROJECT_PATH
from keras.utils import to_categorical

from neuronal_network import NN


def load_data():
    set_path = PROJECT_PATH.joinpath('set')

    images=[]
    labels=[]

    for photo in os.listdir(set_path):
        im = cv2.imread(str(set_path.joinpath(photo)), cv2.IMREAD_COLOR)
        images.append(im)
        labels.append(int('class1' in photo))

    trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, random_state=42)

    Y_train = to_categorical(trainY, num_classes=2)
    Y_test = to_categorical(testY, num_classes=2)

    return np.array(trainX), Y_train, np.array(testX), Y_test


if __name__ == "__main__":
    nn=NN([2,64,10])