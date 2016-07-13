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

    def __init__(self, n_folds=10):
        self.n_folds = n_folds

    def run(self, X_train, y_train, X_test, y_test, profiler):
        skf = StratifiedKFold(y_train, n_folds=self.n_folds,
                              shuffle=True, random_state=123)
        fold = 1
        for train_index, test_index in skf:
            X_train_fold, y_train_fold = [X_train[i] for i in train_index], [y_train[i] for i in train_index]
            X_test_fold, y_test_fold = [X_test[i] for i in test_index], [y_test[i] for i in test_index]
            logger.info('Training on {} instances!'.format(len(train_index)))
            profiler.train(X_train_fold, y_train_fold)
            logger.info('Testing on fold {} with {} instances'.format(
                fold, len(test_index)))
            y_pred_fold = profiler.predict(X_test_fold)
            print_accuracy(y_test_fold, y_pred_fold)
            fold = fold + 1
        if X_test:
            logger.info('Training on {} instances!'.format(len(X_train)))
            profiler.train(X_train, y_train)
            logger.info('Testing on {} instances!'.format(len(X_test)))
            y_pred = profiler.predict(X_test)
            print_confusion_matrix(y_test, y_pred)
            print_accuracy(y_test, y_pred)
