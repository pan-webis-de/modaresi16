from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ToDo: Extract this method to a new module. This can be used by other profilers too!
# ToDo: Add other classifiers
def get_classifier(method='logistic_regression'):
    if 'logistic_regression' == method:
        return LogisticRegression(C=1e3,
                                  tol=0.01,
                                  multi_class='ovr',
                                  solver='liblinear',
                                  n_jobs=-1,
                                  random_state=123)
    if 'random_forest' == method:
        return RandomForestClassifier(n_estimators=250,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=123)

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
