import logging
from sklearn.cross_validation import StratifiedKFold
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.cross_validation import StratifiedKFold


logger = logging.getLogger(__name__)


def print_confusion_matrix(Y_test, Y_pred):
    print '*' * 50
    print 'Confusion Matrix'
    print '*' * 50
    print(ConfusionMatrix(Y_test, Y_pred))
    print '*' * 50


def print_accuracy(Y_test, Y_pred):
    print '+' * 50
    print 'Accuracy: {}'.format(accuracy_score(Y_test, Y_pred))
    print '+' * 50


class SklearnBenchmark():
    def __init__(self, n_folds=3):
        self.n_folds = 3

    def run(self, training_dataset_iterator, test_dataset_iterator, profiler):
        dataset_list = list(training_dataset_iterator)
        X = [xy[0] for xy in dataset_list]
        Y = [xy[1] for xy in dataset_list]
        skf = StratifiedKFold(Y, n_folds=self.n_folds, shuffle=True, random_state=123)
        fold = 1
        for train_index, test_index in skf:
            X_train, Y_train = [X[i] for i in train_index], [Y[i] for i in train_index]
            X_test, Y_test = [X[i] for i in test_index], [Y[i] for i in test_index]
            logger.info('Training on {} instances!'.format(len(train_index)))
            profiler.train(X_train, Y_train)
            logger.info('Testing on fold {} with {} instances'.format(fold, len(test_index)))
            Y_pred = profiler.predict(X_test)
            print_accuracy(Y_test, Y_pred)
            fold = fold + 1
        if test_dataset_iterator:
            test_dataset_list = list(test_dataset_iterator)
            X_test = [xy[0] for xy in test_dataset_list]
            Y_test = [xy[1] for xy in test_dataset_list]
            logger.info('Training on {} instances!'.format(len(X)))
            profiler.train(X, Y)
            logger.info('Testing on {} instances!'.format(len(X_test)))
            Y_pred = profiler.predict(X_test)
            print_confusion_matrix(Y_test, Y_pred)
            print_accuracy(Y_test, Y_pred)
