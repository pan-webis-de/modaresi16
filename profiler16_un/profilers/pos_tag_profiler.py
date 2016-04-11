from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from polyglot.text import Text
from polyglot.text import Text, Word
import logging
from profiler16_un.taggers.polyglot_pos_tagger import PolyglotPOSTagger

import math

# eager instantiation
pos_tagger_en = PolyglotPOSTagger(lang='en')
pos_tagger_nl = PolyglotPOSTagger(lang='nl')
pos_tagger_es = PolyglotPOSTagger(lang='es')


def pos_tags(text='', lang='en'):
    if 'en' == lang:
        return pos_tagger_en.pos_tags(text)
    if 'es' == lang:
        return pos_tagger_es.pos_tags(text)
    if 'nl' == lang:
        return pos_tagger_nl.pos_tags(text)


def tokenize(x):
    return x.split()


def normalize_vector(x):
    x = np.array(x, dtype=np.float64)
    sumvalue = 0
    for i in xrange(0, 16):
        sumvalue += (x[i] * x[i])
    sumvalue = math.sqrt(sumvalue)
    if sumvalue > 0:
        for i in xrange(0, 16):
            x[i] = x[i] / sumvalue
    return x


def get_pos_tags_array():
    return np.array(
        ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
         'SCONJ', 'SYM', 'VERB', 'X'])


def get_pos_tag_distribution(x, language):
    pos_tag_count_dictionary = dict.fromkeys(get_pos_tags_array(), 0)
    for word, tag in pos_tags(text=x, lang=language):
        pos_tag_count_dictionary[tag] += 1
    return pos_tag_count_dictionary


class POSFeatures(BaseEstimator):
    def __init__(self, language='en'):
        self.language = language
        # print("{} {}".format("language:", self.language))

    def get_feature_names(self):
        return get_pos_tags_array

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        distributions = [normalize_vector(get_pos_tag_distribution(doc, self.language).values()) for doc in documents]
        X = np.array(distributions)
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)


# force language to skip language detection step that might cause the POS tagger to load the wrong model
class POSTagProfiler():
    def __init__(self, language='en'):
        logger = logging.getLogger("polyglot.mapping.expansion")
        logger.setLevel(logging.WARNING)
        self.pipeline = Pipeline([('vect', POSFeatures(language=language)),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
