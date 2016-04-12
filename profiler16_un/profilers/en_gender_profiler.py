from ..pipelines.pipelines import avg_spelling_error
from ..pipelines.pipelines import pos_distribution
from ..pipelines.pipelines import word_unigrams
from sklearn.feature_selection import f_classif
from ..pipelines.pipelines import word_bigrams
from ..pipelines.pipelines import char_ngrams
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import chi2
from sklearn.pipeline import FeatureUnion
from ..utils.utils import get_classifier
from sklearn.pipeline import Pipeline


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
