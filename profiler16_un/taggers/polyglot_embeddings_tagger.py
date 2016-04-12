from polyglot.mapping import Embedding


class PolyglotEmbeddingsTagger:
    """Simple wrapper for polyglot embeddings"""

    def __init__(self, lang='en'):
        if 'en' == lang:
            self.embeddings = Embedding.load(
                "/root/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
        if 'es' == lang:
            self.embeddings = Embedding.load(
                "/root/polyglot_data/embeddings2/es/embeddings_pkl.tar.bz2")
        if 'nl' == lang:
            self.embeddings = Embedding.load(
                "/root/polyglot_data/embeddings2/nl/embeddings_pkl.tar.bz2")

    def has_embedding(self, text=''):
        """Returns whether the provided token has a word embedding in the loaded model.
        """
        if not text:
            return False
        return text in self.embeddings
