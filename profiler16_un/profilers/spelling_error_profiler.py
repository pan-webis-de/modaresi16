from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np

from profiler16_un.postprocessors.hunspell_wrapper import HunspellWrapper


def avg_error(tokens):
    if len(tokens) == 0:
        return 0.0
    trueSum = 0
    for token in tokens:
        if is_correct(token):
            trueSum += 1

    # print(1.0 * trueSum / len(tokens))
    return 1.0 * trueSum / len(tokens)


# eager instantiation
hunspell_en = HunspellWrapper(lang='en')
hunspell_nl = HunspellWrapper(lang='nl')
hunspell_es = HunspellWrapper(lang='es')


def is_correct(text='', lang='en'):
    if 'en' == lang:
        return hunspell_en.is_correct(text)
    if 'es' == lang:
        return hunspell_es.is_correct(text)
    if 'nl' == lang:
        return hunspell_nl.is_correct(text)


def tokenize(x):
    return x.split()


class SpellingErrorProfiler():
    def __init__(self, language='en'):
        self.pipeline = Pipeline([('vect', SpellingError(language=language)),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)


class SpellingError(BaseEstimator):
    def __init__(self, language='en'):
        self.language = language
        print("{} {}".format("language:", self.language))

    def get_feature_names(self):
        return np.array(['avg_error_count'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        avg_errors = [avg_error(tokens) for tokens in tokens_list]
        X = np.array([avg_errors]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)
