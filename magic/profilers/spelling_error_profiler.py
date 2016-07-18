from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from magic.postprocessors.hunspell_wrapper import HunspellWrapper
from string import printable


def tokenize(x):
    x = filter(lambda x: x in printable, x)
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
        self.hunspell_en = HunspellWrapper(lang='en')
        self.hunspell_nl = HunspellWrapper(lang='nl')
        self.hunspell_es = HunspellWrapper(lang='es')

    def get_feature_names(self):
        return np.array(['avg_error_count'])

    def fit(self, documents, y=None):
        return self

    def avg_error(self, tokens):
        if len(tokens) == 0:
            return 0.0
        trueSum = 0
        for token in tokens:
            if self.is_correct(token):
                trueSum += 1
        return 1.0 * trueSum / len(tokens)

    def is_correct(self, text='', lang='en'):
        if 'en' == lang:
            return self.hunspell_en.is_correct(text)
        if 'es' == lang:
            return self.hunspell_es.is_correct(text)
        if 'nl' == lang:
            return self.hunspell_nl.is_correct(text)

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        avg_errors = [self.avg_error(tokens) for tokens in tokens_list]
        X = np.array([avg_errors]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)
