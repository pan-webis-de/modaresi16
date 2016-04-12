def punctuation_ngrams():
    pprorcessor = TextCleaner(filter_urls=True, filter_mentions=True, filter_hashtags=True, only_punctuation=True, lowercase=False)
    vectorizer = CountVectorizer(min_df=1, preprocessor=pprcessor, analyzer='char', ngram_range=(10, 10))
    punctuation_ngrams = ('punctuation_ngrams', Pipeline([('vect', vectorizer),
                                                          ('tfidf', TfidfTransformer(sublinear_tf=True)),
                                                          ('scale', Normalizer())]))
    return punctuation_ngrams

