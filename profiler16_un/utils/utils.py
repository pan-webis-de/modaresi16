from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from nltk.corpus import stopwords


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

    if 'gradient_boosting' == method:
        return xgb.XGBClassifier(max_depth=10,
                                 subsample=0.7,
                                 n_estimators=500,
                                 min_child_weight=0.05,
                                 colsample_bytree=0.3,
                                 learning_rate=0.1)


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=50):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]
    for coef, feat in topn:
        print classlabel, feat.encode('utf-8'), coef


def show_most_informative_features(vectorizer, clf, n=50):
    print 'Total number of features: {}'.format(len(vectorizer.get_feature_names()))
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%-15s\t\t%-15s" % (fn_1.encode('utf-8'), fn_2.encode('utf-8'))


def get_stopwords(languages=['english', 'dutch', 'spanish']):
    result = []
    [result.extend(stopwords.words(lang)) for lang in languages]
    return result
