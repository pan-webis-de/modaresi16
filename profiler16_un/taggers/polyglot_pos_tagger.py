from polyglot.tag import POSTagger
from polyglot.text import Text


class PolyglotPOSTagger:
    """Simple wrapper for polyglot pos tagger"""

    def __init__(self, lang='en'):
        self.pos_tagger = POSTagger(lang=lang)

    def pos_tags(self, text=''):
        """Returns an list of tuples of the form (word, POS tag).
        Example:
        ::
            [('At', 'ADP'), ('eight', 'NUM'), ("o'clock", 'NOUN'), ('on', 'ADP'),
            ('Thursday', 'NOUN'), ('morning', 'NOUN')]
        :rtype: list of tuples
        """
        if not text:
            return []
        tagged_words = []
        blob = Text(text)
        try:
            result = self.pos_tagger.annotate(blob.words)
        except:
            print 'Error'
            return []
        for word, t in self.pos_tagger.annotate(blob.words):
            word.pos_tag = t
            tagged_words.append((word, t))
        return tagged_words
