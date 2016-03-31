from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from ..utils.utils import get_classifier


class CharacterNGramProfiler():
    def __init__(self, min_n=1, max_n=1, method='logistic_regression'):
        self.pipeline = Pipeline([('vect', CountVectorizer(min_df=1,
                                                           analyzer='char',
                                                           lowercase=True,
                                                           ngram_range=(min_n, max_n),
                                                           max_features=2000
                                                           )),
                                  ('tfidf', TfidfTransformer(sublinear_tf=False,
                                                             use_idf=False
                                                             )),
                                  ('chi', SelectKBest(chi2, k='all')),
                                  ('method', get_classifier(method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
