import glob
import pandas as pd
import re
from settings.scraper import TARGET_LANGS
import pickle


def label_map(label):
    if label == 'fake':
        return 1
    else:
        return 0
    
    
def dataset_reader(news_path, lang_features=False):
    dataset = pd.DataFrame(columns=['file', 'headline', 'content', 'label'])

    regex_pattern = r"[\n.]"
    for filename in glob.glob(news_path + "*.txt"):
        with open(filename, 'r') as file:
            text = file.read().strip()
            text_splitted = re.split(regex_pattern, text)
            text_splitted = [sentence for sentence in text_splitted if len(sentence) > 0]
            headline = ''
            for index, sentence in enumerate(text_splitted):
                headline += sentence
                if len(headline.split(' ')) >= 5:
                    break
            content = '.'.join(text_splitted[index + 1:])
            news_file = filename.split('/')[-1]
            dataset = dataset.append({'file': news_file,
                                      'headline': headline.strip(),
                                      'content': content.strip(),
                                      'label': news_file.split('.')[1]}, ignore_index=True)

    if lang_features:
        for lang in TARGET_LANGS:
            dataset.insert(len(dataset.columns), lang, 0)

    return dataset


def mult_evidence_reader(path):
    with open(path, 'rb') as file:
        return pickle.load(file)