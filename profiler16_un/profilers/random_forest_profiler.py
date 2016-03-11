from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class RandomForestProfiler():
    def __init__(self):
        # setup logistic regression (Regularization hyperparam C has drastic influence!)
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                                  ('tfidf', TfidfTransformer(sublinear_tf=True)),
                                  ('rf', RandomForestClassifier(n_estimators=500, random_state=123))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
