import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deeppavlov import configs, build_model
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=False)
ACCEPTED_NER_LABELS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']


def preprocessing(text):
    stemmer = WordNetLemmatizer()

    result = text.lower()
    result = re.sub(r'\W', ' ', result)
    result = re.sub(r'[^\x00-\x7f]',r'', result)
    result = result.strip()

    tokens = result.split(' ')
    tokens = [stemmer.lemmatize(token) for token in tokens]
    bigrams = list(nltk.bigrams(tokens))
    bigrams = [''.join(bigram) for bigram in bigrams]
    result = ' '.join(tokens + bigrams)

    return result


def get_ne(text):   
    parse_result = ner_model([text])

    nes = []
    for title, label in zip(parse_result[0], parse_result[1]):
        label_index = 0
        ne = []
        start_index = 0
        in_ne = False
        while label_index < len(label):
            if label[label_index] == 'O':
                if len(ne) > 0:
                    nes.append({
                        'NE': ' '.join(ne),
                        'start_index': start_index,
                        'end_index': label_index - 1,
                    })
                ne = []
                in_ne = False
            else:
                if label[label_index].split('-')[1] in ACCEPTED_NER_LABELS:
                    ne.append(title[label_index])
                    if not in_ne:
                        in_ne = True
                        start_index = label_index

            label_index += 1
            
    if len(ne) > 0:
        nes.append({
                'NE': ' '.join(ne),
                'start_index': start_index,
                'end_index': label_index - 1,
                })

    return nes


#fix not readable symbols on russian
def decode_ru(s):
    try:
        return s.encode('latin1').decode('utf8')
    except:
        return s


def fix_encoding(df):
    df.loc[df.language=='ru', 'headline'] = df[df.language=='ru'].headline.apply(lambda x: decode_ru(x))
    df.loc[df.language=='ru', 'content'] = df[df.language=='ru'].content.apply(lambda x: decode_ru(x))