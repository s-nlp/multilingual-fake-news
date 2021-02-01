from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import liwc
from collections import Counter
import textstat
import string
from nltk.corpus import stopwords
from settings.scraper import TARGET_LANGS
import stanza

nlp = stanza.Pipeline('en', use_gpu=False) # stanza.download('en')
parse, category_names = liwc.load_token_parser('../data/tools/LIWC2015_English.dic')


def append_features(X1, X2):
    if len(X1.index) == 0:
        return X2
    return pd.concat([X1, X2], axis=1)


def tfidf_extractor(dataset):
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X_tfidf = tfidfconverter.fit_transform(dataset['text'].values).toarray()
    features = pd.DataFrame(X_tfidf, columns = tfidfconverter.get_feature_names())
    
    return features


def punct_extractor(dataset):
    column_names = ['punct_'+str(index) for index, punct in enumerate(list(string.punctuation))]

    features = pd.DataFrame(0, index=np.arange(dataset.shape[0]), columns = column_names)
    
    for index, row in dataset.iterrows():
        for index, punct in enumerate(string.punctuation):
            if punct in row['content']:
                features.at[index, 'punct_'+str(index)] += 1
    
    return features


def liwc_extractor(dataset):
    features = pd.DataFrame(0, index=np.arange(dataset.shape[0]), columns = category_names)
    
    for index, row in dataset.iterrows():
        tokens = row['content'].split(' ')
        category_counts = Counter(category for token in tokens for category in parse(token))
        for category, value in category_counts.items():
            features.at[index, category] = value
    
    return features


def readibility_extractor(dataset):
    features = pd.DataFrame(0., index=np.arange(dataset.shape[0]), columns = [
        'flesch_kincaid_grade',
        'flesch_reading_ease',
        'gunning_fog',
        'automated_readability_index',
        'num_char',
        'num_paragraph',
        'max_word_len',
    ])
    
    for index, row in dataset.iterrows():
        features.at[index, 'flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(row['content'])
        features.at[index, 'flesch_reading_ease'] = textstat.flesch_reading_ease(row['content'])
        features.at[index, 'gunning_fog'] = textstat.gunning_fog(row['content'])
        features.at[index, 'automated_readability_index'] = textstat.automated_readability_index(row['content'])
        features.at[index, 'num_char'] = len(row['content'])
        features.at[index, 'num_paragraph'] = len(row['content'].split('\n'))
        features.at[index, 'max_word_len'] = max([len(token) for token in row['content'].split(' ')])
        
    return features


def syntax_preextractor(dataset):
    result = dataset.copy()
    
    result['syntax'] = ''

    for index, row in result.iterrows():
        doc = nlp(row['headline'] + '. ' + row['content'])

        row_feature = ''

        for sentence in doc.sentences:
            for word in sentence.words:
                parent = word.head
                if parent == 0:
                    continue
                parent_pos = sentence.words[parent-1].xpos
                grandparent = sentence.words[parent-1].head
                if grandparent == 0:
                    continue
                grandparent_pos = sentence.words[grandparent-1].xpos
                feature = grandparent_pos + parent_pos + word.lemma
                row_feature += feature + ' '

        result.at[index, 'syntax'] = row_feature
    
    return result


def syntax_extractor(dataset):
    dataset_syntax = syntax_preextractor(dataset)
    
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X_tfidf = tfidfconverter.fit_transform(dataset_syntax['syntax'].values).toarray()
    features = pd.DataFrame(X_tfidf, columns = tfidfconverter.get_feature_names())
    
    return features


def mult_evidence_similarity_extractor(dataset, mult_evidence, n_articles=10):
    column_names = []
    for lang in TARGET_LANGS:
        for i in range(n_articles):
            column_names.append(lang + '_' + str(i) + '_sim')

    features = pd.DataFrame(0., index=np.arange(dataset.shape[0]), columns = column_names)
    features['file'] = dataset['file'].copy()

    for file, evidence in mult_evidence.items():
        for lang, res_articles in evidence.items():
            for i, article in enumerate(res_articles[:n_articles]):
                features.at[dataset.file==file, lang + '_' + str(i) + '_sim'] = article['similarity']

    return features.drop('file', axis=1)


def mult_evidence_rank_extractor(dataset, mult_evidence, n_articles=10):
    column_names = []
    for lang in TARGET_LANGS:
        for i in range(n_articles):
            column_names.append(lang + '_' + str(i) + '_rank')

    features = pd.DataFrame(0., index=np.arange(dataset.shape[0]), columns = column_names)
    features['file'] = dataset['file'].copy()

    for file, evidence in mult_evidence.items():
        for lang, res_articles in evidence.items():
            for i, article in enumerate(res_articles[:n_articles]):
                features.at[dataset.file==file, lang + '_' + str(i) + '_rank'] = article['alexa_rank']

    return features.drop('file', axis=1)