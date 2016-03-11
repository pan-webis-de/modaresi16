from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class LogisticRegressionProfiler():
    def __init__(self):
        # setup logistic regression (Regularization hyperparam C has drastic influence!)
        self.pipeline = Pipeline([('vect', CountVectorizer(min_df=1, ngram_range=(1, 1), max_features=20000)),
                                  ('tfidf', TfidfTransformer(sublinear_tf=True)),
                                  ('chi', SelectKBest(chi2, k='all')),
                                  ('lr', LogisticRegression(C=1e3, tol=0.01, multi_class='ovr', solver='liblinear', random_state=123))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
