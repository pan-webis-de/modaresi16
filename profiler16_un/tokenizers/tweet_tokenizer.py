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

regex_str = [emoticons_str, html_tags_str, mentions_str, hash_tags_str,
             urls_str, numbers_str, words_str, other_words_str, anything_else_str]


class TweetTokenizer(object):
    def __init__(self):
        self.tokens_re = re.compile(
            r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

    def __call__(self, doc):
        return self.tokens_re.findall(doc)
