import pandas as pd
import pickle
import glob
from tqdm import tqdm
from settings.data import *
from tools import *
from settings.similarity import SIMILARITY_BASELINE
from settings.translator import translator


def dataset_preparation():
    dataset = pd.DataFrame(columns=['file', 'headline', 'content', 'label'])

    for filename in glob.glob(LEGIT_PATH + "*.txt"):
        with open(filename, 'r') as file:
            headline = file.readline()
            content = file.read()
            dataset = dataset.append({'file': filename.split('/')[-1],
                                      'headline': headline.strip(),
                                      'content': content.strip(),
                                      'label': 'legit'}, ignore_index=True)

    for filename in glob.glob(FAKE_PATH + "*.txt"):
        with open(filename, 'r') as file:
            headline = file.readline()
            content = file.read()
            dataset = dataset.append({'file': filename.split('/')[-1],
                                      'headline': headline.strip(),
                                      'content': content.strip(),
                                      'label': 'fake'}, ignore_index=True)

    for lang in TARGET_LANGS:
        dataset.insert(len(dataset.columns), lang, 0)

    return dataset


def news_evaluation():
    dataset = dataset_preparation()

    # storing the search results or each news divided by languages
    scraping_results = {}

    for index, news in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        scraping_results[news['file']] = {}

        for lang in TARGET_LANGS:
            scraping_results[news['file']][lang] = []

            similar_count = 0

            translated_news = {'headline': translator.translate(news['headline'], dest=lang).text,
                               'content': translator.translate(news['content'], dest=lang).text}
            request = generate_request(translated_news)
            search_results = extract_search_results(request)

            for search_result in search_results:
                scraping_results[news['file']][lang].append(search_result)

                news_similarity = calculate_similarity(news, search_result, lang)
                scraping_results[news['file']][lang][-1]['similarity'] = news_similarity

                if news_similarity >= SIMILARITY_BASELINE:
                    similar_count += 1

            dataset.at[index, lang] = similar_count

            # saving
            dataset.to_csv('fakenewsdataset_results.tsv', sep='\t', index=False)
            with open('fakenewsdataset_scraping_results.pkl', 'wb') as file:
                pickle.dump(scraping_results, file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    news_evaluation()