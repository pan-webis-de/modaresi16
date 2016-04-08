import logging
from sklearn.cross_validation import StratifiedKFold
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.cross_validation import StratifiedKFold


logger = logging.getLogger(__name__)


class SklearnBenchmark():

    def run(self, training_dataset_iterator, test_dataset_iterator, profiler):
        dataset_list = list(training_dataset_iterator)
        X = [xy[0] for xy in dataset_list]
        Y = [xy[1] for xy in dataset_list]
        skf = StratifiedKFold(Y, n_folds=3, shuffle=True, random_state=123)
        fold = 1
        for train_index, test_index in skf:
            X_train, Y_train = [X[i] for i in train_index], [Y[i] for i in train_index]
            X_test, Y_test = [X[i] for i in test_index], [Y[i] for i in test_index]
            profiler.train(X_train, Y_train)
            Y_pred = profiler.predict(X_test)
            print '*' * 50
            print 'Confusion Matrix -> Fold {}'.format(fold)
            print '*' * 50
            print(ConfusionMatrix(Y_test, Y_pred))
            print '+' * 50
            print 'Accuracy: {}'.format(accuracy_score(Y_test, Y_pred))
            print '*' * 50
            fold = fold + 1
        if test_dataset_iterator:
            test_dataset_list = list(test_dataset_iterator)
            X_test = [xy[0] for xy in test_dataset_list]
            Y_test = [xy[1] for xy in test_dataset_list]
            profiler.train(X, Y)
            Y_pred = profiler.predict(X_test)
            print '*' * 50
            print 'Confusion Matrix (Test Data)'.format(fold)
            print '*' * 50
            print(ConfusionMatrix(Y_test, Y_pred))
            print '+' * 50
            print 'Accuracy: {}'.format(accuracy_score(Y_test, Y_pred))
            print '*' * 50
