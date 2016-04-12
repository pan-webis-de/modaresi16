from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from ..preprocessors.text_cleaner import TextCleaner


def punctuation_ngrams():
    pprorcessor = TextCleaner(filter_urls=True,
                              filter_mentions=True,
                              filter_hashtags=True,
                              only_punctuation=True,
                              lowercase=False)
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=pprcessor,
                                 analyzer='char',
                                 ngram_range=(10, 10))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('punctuation_ngrams', pipeline)

def avg_spelling_error():
    return ('avg_spelling_error', Pipeline([('feature', SpellingError(language=lang))]))

def pos_distribution():
    pipeline = Pipeline([('feature', POSFeatures(language=lang)),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('pos_distribution', pipeline) 
