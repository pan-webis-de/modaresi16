from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np


def tokenize(x):
    return x.split()


def average_token_length(tokens):
    if len(tokens) == 0:
        return 0
    return sum(map(len, tokens)) / float(len(tokens))


def num_tokens(tokens):
    return len(tokens)


class BasicFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(['average_token_length', 'num_tokens'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        average_token_lengths = [average_token_length(tokens) for tokens in tokens_list]
        num_tokenss = [num_tokens(tokens) for tokens in tokens_list]
        X = np.array([average_token_lengths, num_tokenss]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)


class AlekseyProfiler():
    def __init__(self):
        # setup logistic regression (Regularization hyperparam C has drastic influence!)
        self.pipeline = Pipeline([('vect', BasicFeatures()),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
