# write your code here
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Reader:

    def __init__(self, REDUCED_SIZE=None):
        (self.x, self.y), (_, _) = tf.keras.datasets.mnist.load_data()
        assert self.x.shape == (60000, 28, 28)
        assert self.y.shape == (60000,)
        self.x = self.x.reshape(60000, 28 * 28)
        # While we load the whole set of data, the following program only uses 1/10 of it
        if REDUCED_SIZE:
            self.x = self.x[:REDUCED_SIZE, :]
            self.y = tuple(self.y[:REDUCED_SIZE])

        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def printer(self):
        print(f'x_train shape: {self.x_train.shape}')
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_train shape: {len(self.y_train)}')
        print(f'y_test shape: {len(self.y_test)}')
        print('Proportion of samples per class in train set:')
        ds = pd.Series(self.y_train)
        freq = ds.value_counts(normalize=True)
        print(freq)

    def splitter(self, set_size, RANDOM_SEED):
        """Split the set of data into a training and testing domain
        details cf.
        https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/"""
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=set_size, random_state=RANDOM_SEED)


if __name__ == '__main__':
    RANDOM_SEED = 40  # for reproducibility of the results
    set_size = 0.3
    REDUCED_SIZE = 6000
    digit_reader = Reader(REDUCED_SIZE)
    digit_reader.splitter(set_size, RANDOM_SEED)
    digit_reader.printer()
