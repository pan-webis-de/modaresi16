import hunspell


class HunspellWrapper(object):
    def __init__(self, lang='en'):
        if lang == 'en':
            self.hobj = hunspell.HunSpell('/root/hunspell/en_US.dic', '/root/hunspell/en_US.aff')
        elif lang == 'es':
            self.hobj = hunspell.HunSpell('/root/hunspell/es_ANY.dic', '/root/hunspell/es_ANY.aff')
        elif lang == 'nl':
            self.hobj = hunspell.HunSpell('/root/hunspell/nl_NL.dic', '/root/hunspell/nl_NL.aff')
        else:
            raise ValueError('Unsupported language')

    def is_correct(self, text):
        return self.hobj.spell(text)

    def get_suggestion(self, text):
        return self.hobj.suggest(text)[0]
