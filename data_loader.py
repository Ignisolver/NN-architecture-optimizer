from constans_and_types import RAW_DATA_PATH, READY_DATA_PATH


class DataLoader:
    def __init__(self, n_samples=None, test_size=0.7):
        """
        :param n_samples: number of samples from data set
            if default (None) - all samples
        :param test_size: test_size / n_samples e.g.
            default: 0.7
        """
        pass

    def load_training_data(self):
        """
        Load data for training
        :return: X_train, y_train
        """
        pass

    def load_test_data(self):
        """
        Load data for test
        :return: X_test, y_test
        """
        pass