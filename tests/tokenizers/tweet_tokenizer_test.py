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

if __name__ == '__main__':
    unittest.main()
