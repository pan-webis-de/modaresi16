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
import logging
from polyglot.tag import get_pos_tagger
from profiler16_un.taggers.polyglot_pos_tagger import PolyglotPOSTagger
from sklearn.preprocessing import normalize

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


def get_pos_tags_array():
    return np.array(
        ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
         'SCONJ', 'SYM', 'VERB', 'X'])


def average_token_length(tokens):
    if len(tokens) == 0:
        return 0
    return sum(map(len, tokens)) / float(len(tokens))


def num_tokens(tokens):
    return len(tokens)


def get_pos_tag_distribution(x, language):
    # print(language)
    pos_tag_count_dictionary = dict.fromkeys(get_pos_tags_array(), 0)
    for word, tag in pos_tags(text=x, lang=language):
        pos_tag_count_dictionary[tag] += 1
    return pos_tag_count_dictionary
    # print(pos_tag_count_dictionary)


class POSFeatures(BaseEstimator):
    def __init__(self, language='en', absolute_frequency=True, normalize=False):
        self.language = language
        print("{} {}".format("language:", self.language))

    def get_feature_names(self):
        return get_pos_tags_array

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        tokens_list = [tokenize(doc) for doc in documents]
        # distributions = [normalize(get_pos_tag_distribution(doc, self.language).values()) for doc in documents]
        distributions = [get_pos_tag_distribution(doc, self.language).values() for doc in documents]
        X = np.array(distributions)
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)


# force language to skip language detection step that might cause the POS tagger to load the wrong model
class POSTagProfiler():
    def __init__(self, absolute_frequency=True, normalize=False, language='en'):
        logger = logging.getLogger("polyglot.mapping.expansion")
        logger.setLevel(logging.WARNING)

        # text = Text("Ik ben apetrots op je")
        # text.__lang = 'de'
        # for word, tag in text.pos_tags:
        # print(u"{:<16}{:>2}".format(word, tag).encode('utf-8'))
        # print(text.pos_tags)
        # print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
        # for word, tag in text.pos_tags:
        # print(u"{:<16}{:>2}".format(word, tag)

        # print(downloader.supported_languages_table("pos2"))
        self.pipeline = Pipeline([('vect', POSFeatures(language=language)),
                                  ('svm', SVC())])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)


class TextWithFixedLanguage(Text):
    def __init__(self, text, language='en'):
        self.overwrite_language = language
        super(TextWithFixedLanguage, self).__init__(text)
        # self.language.code = language
        # self.language(language)

        # def pos_tagger(self):
        # return super.get_pos_tagger(lang=self.overwrite_language)
