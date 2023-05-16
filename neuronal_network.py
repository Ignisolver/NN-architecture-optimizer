from keras.models import Sequential

from keras.layers import Flatten
from keras.layers import Dense, BatchNormalization
from keras.optimizers import *
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from constans_and_types import MODELS_PATH, BASE_MODEL

class NN:
    def __init__(self, data):
        self.data = data
        self.model = Sequential(BASE_MODEL.layers)
        self.history = History()
        self.acc = 0
        self.loss = 0
        self._create_network()

    def _create_network(self):
        self._add_danse_layers_()
        self._add_last_layer()

    def _add_danse_layers_(self):
        for layer_size in self.data[:-1]:
            self.model.add(BatchNormalization())
            self.model.add(Dense(units=layer_size, activation='relu',
                                 kernel_initializer='he_uniform'))

    def _add_last_layer(self):
        self.model.add(Flatten())
        self.model.add(Dense(2, activation='softmax'))

    def save_model(self):
        self.model.save(MODELS_PATH.joinpath('Ready_base_model'))

    def train_network(self, trainX, trainY, opt=Adam, loss='binary_crossentropy',
                      learning_rate=0.0001, epoch=5):
        self._prepare_model_to_training(opt, loss, learning_rate)

        es = EarlyStopping(monitor='val_loss', patience=5)
        self.history = self.model.fit(trainX, trainY, batch_size=32,
                                      epochs=epoch, verbose=1,
                                      validation_split=0.3,callbacks=[es])

    def _prepare_model_to_training(self, opt, loss, learning_rate):
        self.model.summary()
        self.model.compile(
            loss=loss,
            optimizer=opt(learning_rate=learning_rate),
            metrics=['accuracy']
        )

    def evaluate_network(self, testX, testY):
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
