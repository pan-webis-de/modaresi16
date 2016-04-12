from ..pipelines.pipelines import word_unigrams, word_bigrams, avg_spelling_error, pos_distribution, char_ngrams
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from ..utils.utils import get_classifier


class EnglishGenderProfiler():
    def __init__(self, lang='en', min_n=1, max_n=1, method=None):
        features = FeatureUnion([word_unigrams(), word_bigrams()], n_jobs=1)
        self.pipeline = Pipeline([('features', features),
                                  ('scale', Normalizer()),
                                  #('chi', SelectKBest(f_classif, k=30000)),
                                  ('classifier', get_classifier(method=method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
