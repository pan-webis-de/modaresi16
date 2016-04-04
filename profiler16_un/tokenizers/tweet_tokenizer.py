import re


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

html_tags_str = r'<[^>]+>'

mentions_str = r'(?:@[\w_]+)'

hash_tags_str = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"

urls_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

numbers_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'

words_str = r"(?:[a-z][a-z'\-_]+[a-z])"  # words with - and '

other_words_str = r'(?:[\w_]+)'

anything_else_str = r'(?:\S)'

regex_str = [emoticons_str, html_tags_str]
regex_str.append(mentions_str)
regex_str.append(hash_tags_str)
regex_str.append(urls_str)
regex_str.append(numbers_str)
regex_str.append(words_str)
regex_str.append(other_words_str)
regex_str.append(anything_else_str)


class TweetTokenizer(object):
    def __init__(self, filter_mentions=False, filter_hashtags=False, filter_urls=False):
        self.tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        self.mentions_re = re.compile(r'(' + mentions_str + ')', re.VERBOSE | re.IGNORECASE)
        self.hashtags_re = re.compile(r'(' + hash_tags_str + ')', re.VERBOSE | re.IGNORECASE)
        self.urls_re = re.compile(r'(' + urls_str + ')', re.VERBOSE | re.IGNORECASE)
        self.filter_mentions = filter_mentions
        self.filter_hashtags = filter_hashtags
        self.filter_urls = filter_urls

    def __call__(self, doc):
        tokens = self.tokens_re.findall(doc)
        if self.filter_mentions:
            [tokens.remove(token) for token in self.mentions_re.findall(doc)]
        if self.filter_urls:
            [tokens.remove(token) for token in self.urls_re.findall(doc)]
        if self.filter_hashtags:
            [tokens.remove(token) for token in self.hashtags_re.findall(doc)]
        return tokens
