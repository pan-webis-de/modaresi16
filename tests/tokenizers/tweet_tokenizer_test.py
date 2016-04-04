import unittest
from profiler16_un.tokenizers.tweet_tokenizer import TweetTokenizer


class TestTweetTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = TweetTokenizer()

    def test_when_text_is_empty(self):
        tokens = self.tokenizer('')
        self.assertFalse(tokens)

    def test_tweet(self):
        tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
        tokens = self.tokenizer(tweet)
        self.assertEqual(tokens, ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP'])

    def test_filter_mentions(self):
        tokenizer = TweetTokenizer(filter_mentions=True)
        tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
        tokens = tokenizer(tweet)
        self.assertEqual(tokens, ['RT', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP'])

    def test_filter_urls(self):
        tokenizer = TweetTokenizer(filter_urls=True)
        tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
        tokens = tokenizer(tweet)
        self.assertEqual(tokens, ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', '#NLP'])

    def test_filter_hashtags(self):
        tokenizer = TweetTokenizer(filter_hashtags=True)
        tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
        tokens = tokenizer(tweet)
        self.assertEqual(tokens, ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com'])

    def test_filter_hashtags_urls_mentions(self):
        tokenizer = TweetTokenizer(filter_hashtags=True, filter_mentions=True, filter_urls=True)
        tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
        tokens = tokenizer(tweet)
        self.assertEqual(tokens, ['RT', ':', 'just', 'an', 'example', '!', ':D'])


if __name__ == '__main__':
    unittest.main()
