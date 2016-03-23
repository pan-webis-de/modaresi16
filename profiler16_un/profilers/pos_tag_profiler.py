from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from polyglot.text import Text
from polyglot.downloader import downloader

import polyglot
from polyglot.text import Text, Word
from polyglot.tag import POSTagger


def tokenize(x):
    return x.split()


def average_token_length(tokens):
    if len(tokens) == 0:
        return 0
    return sum(map(len, tokens)) / float(len(tokens))


def num_tokens(tokens):
    return len(tokens)


class POSFeatures(BaseEstimator):
    def __init__(self, test=5):
        print("{} {}".format("test:", test))

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


class POSTagProfiler():
    def __init__(self, absolute_frequency=True, normalize=False):
        blob = """We will meet at eight o'clock on Thursday morning."""
        #  text = Text(blob)
        #print(text.pos_tags)
        # print(downloader.list())
        # downloader._set_download_dir("/var/www/polyglot_models")
        # print(downloader.default_download_dir())

        # downloader._download_dir("/var/www/polyglot_models")
        # print(downloader.list(show_packages=False).encode('utf-8'))
        text = Text("Ik ben apetrots op je")
        test = POSTagger("nl")
        print(text.pos_tags)
        # print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
        # for word, tag in text.pos_tags:
        # print(u"{:<16}{:>2}".format(word, tag)

        # print(downloader.supported_languages_table("pos2"))
        self.pipeline = Pipeline([('vect', POSFeatures(6)),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
