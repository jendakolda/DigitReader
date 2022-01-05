# write your code here
import numpy as np
import tensorflow as tf


class Reader:

    def __init__(self):
        (self.x_train, self.y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        assert self.x_train.shape == (60000, 28, 28)
        # assert x_test.shape == (10000, 28, 28)
        assert self.y_train.shape == (60000,)
        # assert y_test.shape == (10000,)
        # self.flat = np.empty([self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2]], dtype=int)
        self.flat = self.x_train.reshape(60000, 28 * 28)

    def printer(self):
        print(f'Classes: {list(set(self.y_train))}')
        print(f'Features\' shape: {self.flat.shape}')
        print(f'Target\'s shape: {self.y_train.shape}')
        print(f'min: {self.flat.min()}, max: {self.flat.max()}')


if __name__ == '__main__':
    digit_reader = Reader()
    digit_reader.printer()




