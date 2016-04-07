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


class EnglishGenderProfiler():
    def __init__(self, lang='en', min_n=1, max_n=1, method='logistic_regression'):
        self.ngram_tokens = ('ngram_tokens', CountVectorizer(min_df=1,
                                                             lowercase=True,
                                                             ngram_range=(1, 2),
                                                             tokenizer=TweetTokenizer(filter_urls=True)
                                                             )
                             )
        ngram_chars = ('char_ngrams', Pipeline([
                                                 ('vect2', CountVectorizer(min_df=1,
                                                                           analyzer='char',
                                                                           lowercase=True,
                                                                           ngram_range=(3, 3), max_features=20000)),
                                               ]
                                              )
                      )

        self.pipeline = Pipeline([('features', FeatureUnion([self.ngram_tokens])),
                                  ('tfidf', TfidfTransformer(sublinear_tf=True)),
                                  ('chi', SelectPercentile(chi2, percentile=95)),
                                  ('lr', LogisticRegression(C=1e3,
                                                            tol=0.01,
                                                            multi_class='ovr',
                                                            solver='liblinear',
                                                            random_state=123
                                                            ))
                                 ])


    def most_informative_feature_for_class(self, vectorizer, classifier, classlabel, n=50):
        labelid = list(classifier.classes_).index(classlabel)
        feature_names = vectorizer.get_feature_names()
        topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]
        for coef, feat in topn:
            print classlabel, feat.encode('utf-8'), coef

    def show_most_informative_features(self, vectorizer, clf, n=20):
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1.encode('utf-8'), coef_2, fn_2.encode('utf-8'))


    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)
        self.show_most_informative_features(self.pipeline.named_steps['features'], self.pipeline.named_steps['lr'])

    def predict(self, X):
        return self.model.predict(X)
