import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class Reader:
    """A ML program to recognize a handwritten digits
    utilizes keras dataset (part of tensorflow library) for input"""

    def __init__(self, model, REDUCED_SIZE=None):
        self.model = model

        (self.x, self.y), (_, _) = tf.keras.datasets.mnist.load_data()
        assert self.x.shape == (60000, 28, 28)
        assert self.y.shape == (60000,)
        self.x = self.x.reshape(60000, 28 * 28)
        # While we load the whole set of data, the following program only uses 1/10 of it
        if REDUCED_SIZE:
            self.x = self.x[:REDUCED_SIZE, :]
            self.y = tuple(self.y[:REDUCED_SIZE])

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.score = None

    def splitter(self, set_size, RANDOM_SEED):
        """Split the set of data into a training and testing domain
        details cf.
        https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/"""
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=set_size, random_state=RANDOM_SEED)

    def fit_predict_eval(self):
        """fit the model, make a prediction, calculate accuracy and save it to score"""
        mod = self.model
        mod.fit(self.X_train, self.y_train)
        y_pred = mod.predict(self.X_test)
        self.score = accuracy_score(self.y_test, y_pred)
        print(f'Model: {self.model}\nAccuracy: {round(self.score, 4)}\n')


if __name__ == '__main__':
    RANDOM_SEED = 40  # for reproducibility of the results
    set_size = 0.3
    REDUCED_SIZE = 6000
    models = (KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression(), RandomForestClassifier())
    results = dict()

    for mod in models:
        digit_reader = Reader(mod, REDUCED_SIZE)
        digit_reader.splitter(set_size, RANDOM_SEED)
        digit_reader.fit_predict_eval()
        results[digit_reader.model] = digit_reader.score

    best_acc = max(results, key=results.get)
    print(f'The answer to the question: {best_acc} - {round(results[best_acc], 3)}')
