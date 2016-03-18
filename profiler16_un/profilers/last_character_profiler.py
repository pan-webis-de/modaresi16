from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import datetime
import array
from collections import Mapping, defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.externals.six.moves import xrange
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.hashing import FeatureHasher
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.utils import deprecated
from sklearn.utils.fixes import frombuffer_empty, bincount
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.feature_extraction.text import CountVectorizer

class LastCharacterProfiler():

    def __init__(self, lastCharLength = 1):
        print("{} {}".format("lastCharLength", lastCharLength))
        self.pipeline = Pipeline([('vect', CountVectorizerLastCharacter(min_df=1,
                                                           analyzer='last_chars',
                                                           lowercase=True,
                                                           lastCharLength = lastCharLength,
                                                           max_features=2000,
                                                           )),
                                  ('tfidf', TfidfTransformer(sublinear_tf=False,
                                                             use_idf=False
                                                             )),
                                  ('chi', SelectKBest(chi2, k='all')),
                                  ('method', RandomForestClassifier(n_estimators=250,
                                                bootstrap=False,
                                                n_jobs=-1,
                                                random_state=123))
                                  ])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)




class VectorizerMixinLastChar(VectorizerMixin):
    def _last_chars(self, text_document, length):
        text_document = self._white_spaces.sub(" ", text_document)
        ngrams = []
        for w in text_document.split():
            wLength = len(w)
            if wLength >= length:
                ngrams.append(w[(wLength-length):wLength+length])
        return ngrams



    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'last_chars':
            return lambda doc: self._last_chars(
                preprocess(self.decode(doc)), self.lastCharLength)

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

class CountVectorizerLastCharacter(CountVectorizer, VectorizerMixinLastChar):


    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 lastCharLength = 1,
                 vocabulary=None, binary=False, dtype=np.int64):

        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.lastCharLength = lastCharLength
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df of min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
