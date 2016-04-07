from tweet_tokenizer import TweetTokenizer
from ..taggers.polyglot_pos_tagger import PolyglotPOSTagger


class POSTweetTokenizer(object):
    def __init__(self, lang='en'):
        self.lang = lang
        self.tweet_tokenizer = TweetTokenizer(filter_mentions=True, filter_urls=True, filter_hashtags=True)
        self.tagger = PolyglotPOSTagger(lang=lang)

    def __call__(self, doc):
        tokens = self.tweet_tokenizer(doc)
        return [tag for (word, tag) in self.tagger.pos_tags(tokens=tokens)]
