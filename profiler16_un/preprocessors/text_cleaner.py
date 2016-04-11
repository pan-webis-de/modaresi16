# -*- coding: utf-8 -*-
import regex
import unicodedata
import re


class TextCleaner(object):
    def __init__(self, filter_mentions=False, filter_hashtags=False,
                 filter_urls=False, filter_non_latin=False,
                 lowercase=False, alphabetic=False, strip_accents=False, only_punctuation=False, filter_rt=False):
        self.filter_mentions = filter_mentions
        self.filter_hashtags = filter_hashtags
        self.filter_urls = filter_urls
        self.filter_non_latin = filter_non_latin
        self.lowercase = lowercase
        self.alphabetic = alphabetic
        self.strip_accents = strip_accents
        self.only_punctuation = only_punctuation
        self.filter_rt = filter_rt

    def __call__(self, doc):
        if self.lowercase:
            doc = doc.lower()
        if self.filter_urls:
            doc = regex.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', u'', doc)
        if self.filter_mentions:
            doc = regex.sub(r'(?:@[\w_]+)', u'', doc)
        if self.filter_hashtags:
            doc = regex.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", u'', doc)
        if self.filter_rt:
            doc = regex.sub(r" rt ", u'', doc)
            doc = regex.sub(r" RT ", u'', doc)
        if self.strip_accents:
            nkfd_form = unicodedata.normalize('NFKD', doc)
            doc = nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
        if self.filter_non_latin:
            doc = regex.sub(r'[\u0627-\u064a]', u'', doc)
            doc = regex.sub(r'[\u0600-\u06FF]', u'', doc)
        if self.alphabetic:
            doc = regex.sub("[^a-zA-ZÀ-ÿ']+", " ", doc)
        if self.only_punctuation:
            doc = regex.sub("[\w]", " ", doc)
        return doc.strip()
