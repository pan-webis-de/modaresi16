import logging
from sklearn.cross_validation import StratifiedKFold
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score
import pandas as pd


pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', '{:16.4f}'.format)


logger = logging.getLogger(__name__)


class SklearnBenchmark():

    def run(self, dataset_iterator, profiler, metric, pred_profile, true_profile, n_log_freq=1):
        dataset_list = list(dataset_iterator)
        X = [xy[0] for xy in dataset_list]
        Y = [xy[1] for xy in dataset_list]
        skf = [fold for fold in StratifiedKFold(Y, n_folds=2, shuffle=True, random_state=123)]
        train_index, test_index = skf[0]
        X_train, Y_train = [X[i] for i in train_index], [Y[i] for i in train_index]
        X_test, Y_test = [X[i] for i in test_index], [Y[i] for i in test_index]
        profiler.train(X_train, Y_train)
        print '*' * 50
        print 'Confusion Matrix'
        print '*' * 50
        print(ConfusionMatrix(Y_test, profiler.predict(X_test)))
        print '*' * 50
        print 'Accuracy: {}'.format(accuracy_score(Y_test, profiler.predict(X_test)))
        print '*' * 50
