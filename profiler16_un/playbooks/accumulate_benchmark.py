from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score


class AccumulateBenchmark():

    def run(self, dataset_iterator, profiler, metric, pred_profile, true_profile, n_log_freq=1):
        y_true = []
        y_pred = []
        for idx, xy in enumerate(dataset_iterator):
            actual = pred_profile(profiler, xy[0])
            y_pred.append(unicode(actual))
            expected = true_profile(xy[1])
            y_true.append(unicode(expected))
        print '*' * 50
        print 'Confusion Matrix'
        print '*' * 50
        print(ConfusionMatrix(y_true, y_pred))
        print '*' * 50
        print 'Accuracy: {}'.format(accuracy_score(y_true, y_pred))
        print '*' * 50
