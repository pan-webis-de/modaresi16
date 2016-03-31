from profiler16_un.taggers.polyglot_pos_tagger import PolyglotPOSTagger


pos_tagger_en = PolyglotPOSTagger(lang='en')
pos_tagger_nl = PolyglotPOSTagger(lang='nl')
pos_tagger_es = PolyglotPOSTagger(lang='es')


def pos_tags(text='', lang='en'):
    if 'en' == lang:
        return pos_tagger_en.pos_tags(text)
    if 'es' == lang:
        return pos_tagger_es.pos_tags(text)
    if 'nl' == lang:
        return pos_tagger_nl.pos_tags(text)
