from magic.profilers.spelling_error_profiler import SpellingError
from magic.features.punctuation_features import PunctuationFeatures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from ..preprocessors.text_cleaner import TextCleaner
from sklearn.preprocessing import Normalizer
from ..utils.utils import get_stopwords
from sklearn.pipeline import Pipeline


def avg_spelling_error(lang=None):
    pipeline = Pipeline([('feature', SpellingError(language=lang)),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('avg_spelling_error', pipeline)


def punctuation_features():
    pipeline = Pipeline([('feature', PunctuationFeatures()),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('punctuation_features', pipeline)


def word_unigrams():
    preprocessor = TextCleaner(lowercase=True,
                               filter_urls=True,
                               filter_mentions=True,
                               filter_hashtags=True,
                               alphabetic=True,
                               strip_accents=True,
                               filter_rt=True)
    vectorizer = CountVectorizer(min_df=2,
                                 stop_words=get_stopwords(),
                                 preprocessor=preprocessor,
                                 ngram_range=(1, 1))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_unigrams', pipeline)


def word_bigrams():
    preprocessor = TextCleaner(lowercase=True,
                               filter_urls=True,
                               filter_mentions=True,
                               filter_hashtags=True,
                               alphabetic=True,
                               strip_accents=True,
                               filter_rt=True)
    pipeline = Pipeline([('vect', CountVectorizer(preprocessor=preprocessor,
                                                  ngram_range=(2, 2))),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_bigrams', pipeline)


def char_ngrams():
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=TextCleaner(filter_urls=True,
                                                          filter_mentions=True,
                                                          filter_hashtags=True,
                                                          lowercase=False),
                                 analyzer='char_wb',
                                 ngram_range=(4, 4))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('char_ngrams', pipeline)
