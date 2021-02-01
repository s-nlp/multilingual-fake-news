import spacy
lemmatizers = {
    'en': spacy.load('en'),
    'fr': spacy.load('fr'),  # to load language use 'python -m spacy download fr'
    'de': spacy.load('de'),
    'es': spacy.load('es'),
    # git clone -b v2.1 https://github.com/buriy/spacy-ru.git && cp -r ./spacy-ru/ru2/. /ru2
    'ru': spacy.load('../spacy-ru/ru2'),  # set path to the lib
}

TIMEOUT = 5

TARGET_LANGS = ['en', 'fr', 'de', 'es', 'ru']

FAKE_FLAGS = {
    'en': {'None', 'fake', 'lie', 'false', 'falsity', 'falsely', 'rumor', 'disprove'},
    'fr': {'None', 'faux', 'mensonge', 'faux', 'fausseté', 'rumeur', 'réfuter'},
    'de': {'None', 'falsch', 'Lüge', 'Gerücht', 'widerlegen'},
    'es': {'None', 'falsa', 'mentira', 'falso', 'falsedad', 'rumor', 'refutar'},
    'ru': {'None', 'фальшивка', 'ложь', 'фальш', 'ложный', 'ложные', 'слух', 'опровергать'}
}
for lang in FAKE_FLAGS:
    lang_flags = FAKE_FLAGS[lang].copy()
    for flag in FAKE_FLAGS[lang]:
        lemma = lemmatizers[lang](flag)[0].lemma_
        lang_flags.add(lemma)
    FAKE_FLAGS[lang] = lang_flags

FORBIDDEN_TYPES = ['pdf', 'txt', '.gz', 'rar', 'doc', 'xml']
NUM_RESULT_PAGES = 2