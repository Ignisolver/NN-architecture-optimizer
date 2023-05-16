from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import load_model
from keras.optimizers import *
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from constans_and_types import MODELS_PATH


class NN:
    def __init__(self, data):
        self.data = data
        self.model = Sequential()
        self.history = History()
        self.acc = 0
        self.loss = 0

    def add_layers(self, pretrain=False):
        if not pretrain:
            model = load_model(MODELS_PATH.joinpath('Brest CNN 2.h5'))
            self.model = Sequential(model.layers[:14])
            for layer in self.model.layers:
                layer.trainable = False

        for i in range(len(self.data)):
            self.model.add(BatchNormalization())
            self.model.add(Dense(units=self.data[i], activation='relu',
                                 kernel_initializer='he_uniform'))

    def save_model(self):
        self.model.save(MODELS_PATH.joinpath('NN_model'))

    def train_network(self,trainX,trainY, opt=Adam, loss='binary_crossentropy',
                      learning_rate=0.0001,epoch=5,pretrain=False):
        self.model.add(Flatten())
        self.model.add(Dense(2,activation='softmax'))
        self.model.summary()
        self.model.compile(
            loss=loss,
            optimizer=opt(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        # tutaj zalezy czy mamy juz podzial na train i test.
        es = EarlyStopping(monitor='val_loss', patience=5)
        self.history = self.model.fit(trainX, trainY, batch_size=32,
                                      epochs=epoch, verbose=1,
                                      validation_split=0.3,callbacks=[es])
        if pretrain:
            self.save_model()

    def evaluate_network(self,testX,testY):
        # tutaj zalezy czy mamy juz podzial na train i test.
        stats = self.model.evaluate(testX, testY, batch_size=32, verbose=1)
        self.loss = stats[0]
        self.acc = stats[1]

    def print(self):
        return self.model.summary()

    def plot_history(self):
        plt.title('Classification Accuracy')
        plt.plot(self.history.history['accuracy'], color='blue', label='train')
        plt.plot(self.history.history['val_accuracy'], color='orange', label='test')
        plt.show()
