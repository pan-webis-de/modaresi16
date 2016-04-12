from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from profiler16_un.taggers.polyglot_embeddings_tagger import PolyglotEmbeddingsTagger


def tokenize(x):
    return x.split()


class EmbeddingsProfiler():

    def __init__(self, language='en'):
        self.pipeline = Pipeline([('vect', EmbeddingsCounter(language=language)),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)


class EmbeddingsCounter(BaseEstimator):

    def __init__(self, language='en'):
        self.language = language
        self.polyglot_embedding_en = PolyglotEmbeddingsTagger(lang='en')
        self.polyglot_embedding_nl = PolyglotEmbeddingsTagger(lang='nl')
        self.polyglot_embedding_es = PolyglotEmbeddingsTagger(lang='es')
        print("{} {}".format("language:", self.language))

    def get_feature_names(self):
        return np.array(['avg_embedding_count'])

    def fit(self, documents, y=None):
        return self

    def has_embedding(self, text='', lang='en'):
        if 'en' == lang:
            return self.polyglot_embedding_en.has_embedding(text)
        if 'es' == lang:
            return self.polyglot_embedding_nl.has_embedding(text)
        if 'nl' == lang:
            return self.polyglot_embedding_es.has_embedding(text)

    def avg_embedding_count(self, tokens):
        if len(tokens) == 0:
            return 0.0
        trueSum = 0
        for token in tokens:
            if self.has_embedding(token):
                trueSum += 1

        return 1.0 * trueSum / len(tokens)

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        avg_embeddings = [self.avg_embedding_count(
            tokens) for tokens in tokens_list]
        X = np.array([avg_embeddings]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)
