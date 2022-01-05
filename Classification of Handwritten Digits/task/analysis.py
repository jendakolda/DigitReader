# write your code here
import numpy as np
import tensorflow as tf


class Reader:

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)
        self.x_train, self.y_train = x_train, y_train
        self.flat = np.empty([self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2]], dtype=int)

    def flatten_features(self):
        for num in range(self.x_train.shape[0]):
            matrix = self.x_train[num, :, :]
            self.flat[num, :] = matrix.flatten()

    def printer(self):
        word = " ".join([str(i) for i in set(self.y_train)])
        print(f'Classes: [{word}]')
        print(f'Features\' shape: {self.flat.shape}')
        print(f'Target\'s shape: {self.y_train.shape}')
        print(f'min: {self.flat.min()}, max: {self.flat.max()}')


if __name__ == '__main__':
    digit_reader = Reader()
    digit_reader.flatten_features()
    digit_reader.printer()




