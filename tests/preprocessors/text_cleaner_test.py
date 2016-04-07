#!/usr/bin/env python
# -*- coding: utf-8 -*-
from profiler16_un.preprocessors.text_cleaner import TextCleaner
import unittest


class TestTextCleaner(unittest.TestCase):

    def test_when_text_is_empty(self):
        cleaner = TextCleaner()
        self.assertEqual(cleaner(u''), u'')

    def test_default_cleaner_does_nothing(self):
        cleaner = TextCleaner()
        self.assertEqual(cleaner(u'http://www.google.de @pasmod'), u'http://www.google.de @pasmod')

    def test_filtering_non_latin_characters(self):
        cleaner = TextCleaner(filter_non_latin=True)
        self.assertEqual(cleaner(u'Hello عیت world'), u'Hello  world')

    def test_filtering_urls(self):
        cleaner = TextCleaner(filter_urls=True)
        self.assertEqual(cleaner(u'Hello http://www.google.de bye'), u'Hello  bye')

    def test_filtering_mentions(self):
        cleaner = TextCleaner(filter_mentions=True)
        self.assertEqual(cleaner(u'Hello @pasmod http://www.google.de'), u'Hello  http://www.google.de')

    def test_filtering_hashtags(self):
        cleaner = TextCleaner(filter_hashtags=True)
        self.assertEqual(cleaner(u'Hello @pasmod #httpe'), u'Hello @pasmod')

    def test_filtering_hashtags(self):
        cleaner = TextCleaner(filter_hashtags=True)
        self.assertEqual(cleaner(u'Hello @pasmod #httpe'), u'Hello @pasmod')
    


if __name__ == '__main__':
    unittest.main()
