from genetic_algorithm.net_data import NetworkData
from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import History
import matplotlib.pyplot as plt


class NN:
    def __init__(self, data: NetworkData):
        self.data = data
        self.model = Sequential()
        self.history = History()
        self.acc = 0
        self.loss = 0

    def add_layer(self, activation_func='relu', kernel_size=3, filters=8, first=False):
        if first:
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding='same',
                                  input_shape=(50, 50, 3)))
            self.model.add(MaxPool2D(pool_size=kernel_size - 1, strides=kernel_size - 1))
        else:
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding='same'))
            self.model.add(MaxPool2D(pool_size=kernel_size - 1, strides=kernel_size - 1))

    def train_network(self, opt=Adam, loss='binary_crossentropy', learning_rate=0.001,epoch=5):
        self.model.add(Flatten())
        self.model.add(Dense(units=2, activation='softmax'))

        self.model.summary()
        self.model.compile(
            loss=loss,
            optimizer=opt(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        # tutaj zalezy czy mamy juz podzial na train i test.

        self.history = self.model.fit(trainX1, trainY, batch_size=32, epochs=epoch, verbose=2)

    def evaluate_network(self):
        # tutaj zalezy czy mamy juz podzial na train i test.
        stats = self.model.evaluate(testX1, testY, batch_size=32, verbose=2)
        self.loss = stats[0]
        self.acc = stats[1]
        return self.acc

    def plot_history(self):
        plt.title('Classification Accuracy')
        plt.plot(self.history.history['accuracy'], color='blue', label='train')
        plt.plot(self.history.history['val_accuracy'], color='orange', label='test')
        plt.show()
