from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ..tokenizers.tweet_tokenizer import TweetTokenizer
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


tc = TextCleaner(lowercase=True,
                 filter_urls=True,
                 filter_mentions=True,
                 filter_hashtags=True,
                 alphabetic=True,
                 strip_accents=True)


class EnglishGenderProfiler():
    def __init__(self, lang='en', min_n=1, max_n=1, method=None):
        unigrams = ('unigrams', CountVectorizer(min_df=1,
                                                stop_words=get_stopwords(),
                                                preprocessor=tc,
                                                ngram_range=(1, 1)
                                                ))
        bigrams = ('bigrams', CountVectorizer(min_df=1,
                                              tokenizer=TweetTokenizer(),
                                              preprocessor=tc,
                                              ngram_range=(2, 2)
                                              ))
        ngram_chars = ('char_ngrams', Pipeline([
                                               ('vect2', CountVectorizer(min_df=1,
                                                                         analyzer='char',
                                                                         lowercase=True,
                                                                         ngram_range=(3, 3), max_features=20000)),
                                               ]))

        self.pipeline = Pipeline([('features', FeatureUnion([unigrams, bigrams])),
                                  ('tfidf', TfidfTransformer(sublinear_tf=True)),
                                  # ('chi', SelectKBest(f_classif, k=30000)),
                                  # ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
                                  # ('pca', TruncatedSVD(n_components=3000, random_state=42)),
                                  # ('hi', RFECV(estimator=get_classifier(method), step=1000,
                                  #                  scoring='accuracy', verbose=1)),
                                  ('lr', get_classifier(method=method))])


    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)
        show_most_informative_features(self.pipeline.named_steps['features'], self.pipeline.named_steps['lr'])

    def predict(self, X):
        return self.model.predict(X)
