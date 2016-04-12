from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ..tokenizers.tweet_tokenizer import TweetTokenizer
from ..tokenizers.lemma_tokenizer import LemmaTokenizer
from sklearn.pipeline import FeatureUnion
from ..utils.utils import get_classifier
from ..utils.utils import show_most_informative_features
from ..utils.utils import get_stopwords
from ..preprocessors.text_cleaner import TextCleaner
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import linear_model, decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from profiler16_un.profilers.spelling_error_profiler import SpellingError
from profiler16_un.profilers.pos_tag_profiler import POSFeatures


def punctuation_ngrams():
    pprorcessor = TextCleaner(filter_urls=True,
                              filter_mentions=True,
                              filter_hashtags=True,
                              only_punctuation=True,
                              lowercase=False)
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=pprcessor,
                                 analyzer='char',
                                 ngram_range=(10, 10))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('punctuation_ngrams', pipeline)


def avg_spelling_error(lang=None):
    return ('avg_spelling_error', Pipeline([('feature', SpellingError(language=lang))]))


def pos_distribution():
    pipeline = Pipeline([('feature', POSFeatures(language=lang)),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('pos_distribution', pipeline) 


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
    pipeline = Pipeline([('vect', CountVectorizer(preprocessor=preprocessor, ngram_range=(2, 2))),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_bigrams', pipeline)
