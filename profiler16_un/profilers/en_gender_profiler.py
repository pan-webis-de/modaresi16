from ..pipelines.pipelines import avg_spelling_error
from ..pipelines.pipelines import word_unigrams
from ..pipelines.pipelines import word_bigrams
from ..pipelines.pipelines import char_ngrams
from ..pipelines.pipelines import avg_embeddings_count
from ..pipelines.pipelines import punctuation_features
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from ..utils.utils import get_classifier
from sklearn.pipeline import Pipeline


class EnglishGenderProfiler():
    def __init__(self, lang=None, method=None, features=None):
        fs = []
        if 'unigram' in features:
            fs.append(word_unigrams())
        if 'bigram' in features:
            fs.append(word_bigrams())
        if 'spelling' in features:
            fs.append(avg_spelling_error(lang=lang))
        if 'embedding' in features:
            fs.append(avg_embeddings_count(lang=lang))
        if 'punctuation' in features:
            fs.append(punctuation_features())
        if 'char' in features:
            fs.append(char_ngrams())

        fu = FeatureUnion(fs, n_jobs=1)
        self.pipeline = Pipeline([('features', fu),
                                  ('scale', Normalizer()),
                                  ('classifier', get_classifier(method=method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
