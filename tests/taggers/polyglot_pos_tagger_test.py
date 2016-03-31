import unittest
from profiler16_un.taggers.polyglot_pos_tagger import PolyglotPOSTagger


class TestPolyglotPOSTagger(unittest.TestCase):

    def setUp(self):
        self.pos_tagger = PolyglotPOSTagger(lang='en')

    def test_when_text_is_empty(self):
        tags = self.pos_tagger.pos_tags('')
        self.assertFalse(tags)

    def test_if_language_detection_is_ignored(self):
        """This test assumes that no German models are available!"""
        tags = self.pos_tagger.pos_tags('Dieser Text ist auf Deutsch')
        self.assertTrue(tags)

if __name__ == '__main__':
    unittest.main()
