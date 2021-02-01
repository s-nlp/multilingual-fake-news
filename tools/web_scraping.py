import urllib.request, sys, re
import xmltodict, json

import os
from googlesearch import search
from contextlib import contextmanager
import re
import signal
import requests
from bs4 import BeautifulSoup
from scipy.spatial import distance
from settings.diffbot import client
from settings.scraper import *
from settings.similarity import bert_embedding
from settings.translator import translator
import pandas as pd
import pickle
from tqdm import tqdm
from tools import dataset_reader


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        

def get_alexa_rank(link):
    try:
        xml = urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url={}'.format(link)).read()
        result= xmltodict.parse(xml)
         
        data = json.dumps(result).replace("@","")
        data_tojson = json.loads(data)
        url = data_tojson["ALEXA"]["SD"][1]["POPULARITY"]["URL"]
        rank = int(data_tojson["ALEXA"]["SD"][1]["POPULARITY"]["TEXT"])
    except:
        print('Link issue.')
        rank = sys.maxsize
    
    return rank


def generate_request_simple(news):
    return news['headline']


def generate_request_cleaned(news):
    return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", news['headline'])


def calculate_sent_embd(sentence):
    if len(sentence) == 0:
        return 0
    embeddings = bert_embedding([sentence.lower()], 'avg')[0][1]
    return sum(embeddings) / len(sentence)


def extract_search_results(request, date_from=None, date_to=None):
    # search_results = google.search(request, NUM_RESULT_PAGES)
    
    result_news = []
    
    try:
        search_results = search(request, num_results=NUM_RESULT_PAGES * 10, date_from=date_from, date_to=date_to)

        for search_result in search_results:
            search_news_headline = ""
            search_news_content = ""

            search_link = search_result  # .link
            print(search_link)
            if search_link is None:
                continue
            if search_link[-3:] in FORBIDDEN_TYPES:
                continue
            try:
                try:
                    with time_limit(TIMEOUT):
                        response = requests.get(search_link)
                except TimeoutException as e:
                    print("Timed out!")
                    continue

                # scraping content
                search_news_content = client.article(search_link)['objects'][0]['text']
                # scraping headline
                soup = BeautifulSoup(response.text)
                res_meta = soup.find_all('meta', property="og:title")
                res_title = soup.find_all('title')
                if not (len(res_meta) > 0 or len(res_title) > 0):
                    continue
                if len(res_meta) > 0:
                    search_news_headline = res_meta[0].attrs['content']
                elif len(res_title) > 0:
                    search_news_headline = res_title[0].contents[0]

                search_link_rank = get_alexa_rank(search_link)
                ############################

                result_news.append({'url': search_link,
                                    'headline': search_news_headline.strip(),
                                    'content': search_news_content.strip(),
                                    'alexa_rank': search_link_rank})

            except Exception as e:
                print(e)

    except Exception as e:
        print(e)
                
    return result_news


def calculate_similarity(news1, news2, lang):
    # peprocessing
    clean_news1_headline = re.sub(r"[^a-zA-Z0-9]+", ' ', news1['headline']).lower()
    clean_news1_content = re.sub(r"[^a-zA-Z0-9]+", ' ', news1['content']).lower()

    clean_news2_headline = re.sub(r"[^a-zA-Z0-9]+", ' ', news2['headline']).lower()
    clean_news2_content = re.sub(r"[^a-zA-Z0-9]+", ' ', news2['content']).lower()

    # if there is any indicator of fake
    if len(FAKE_FLAGS[lang].intersection(set(clean_news2_headline))) > 0:
        return 0

    # embeddings calculation
    embd_news1 = 0.5 * calculate_sent_embd(clean_news1_headline) + 0.5 * calculate_sent_embd(clean_news1_content)
    embd_news2 = 0.5 * calculate_sent_embd(clean_news2_headline) + 0.5 * calculate_sent_embd(clean_news2_content)

    return 1 - distance.cosine(embd_news1, embd_news2)


def multilingual_evidence_scraping(news_path, save_to):
    """
    news_path - the path where all txt files with news are;
    label - legit or fake;
    save_to - where to save the results.
    """
    dataset = dataset_reader(news_path)

    # storing the search results or each news divided by languages
    scraped_results = {}

    for index, news in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        scraped_results[news['file']] = {}

        print(news['headline'])

        for lang in TARGET_LANGS:
            scraped_results[news['file']][lang] = []

            try:
                translated_news = {'headline': translator.translate(news['headline'], dest=lang),
                                   'content': translator.translate(news['content'], dest=lang)}
            except:
                print('Translation issue.')
                translated_news = {'headline': news['headline'],
                                   'content': news['content']}

            request = generate_request_cleaned(translated_news)
            search_results = extract_search_results(request, date_from='1/1/2018', date_to='1/1/2019')

            for search_result in search_results:
                scraped_results[news['file']][lang].append(search_result)

                news_similarity = calculate_similarity(news, search_result, lang)
                scraped_results[news['file']][lang][-1]['similarity'] = news_similarity

            # saving scraped multilingual evidence
            with open(save_to + '.pkl', 'wb') as file:
                pickle.dump(scraped_results, file, pickle.HIGHEST_PROTOCOL)


def manual_news_scraping(save_fake_to, save_legit_to):
    fake_news = [
        'Lottery winner arrested for dumping $200,000 of manure on ex-boss’ lawn',
        'Woman sues Samsung for $1.8M after cell phone gets stuck inside her vagina',
        'BREAKING: Michael Jordan Resigns From The Board At Nike-Takes ’Air Jordans’ With Him',
        'Donald Trump Ends School Shootings By Banning Schools',
        'New mosquito species discovered that can get you pregnant with a single bite',
        'Obama Announces Bid To Become UN Secretary General',
        'Lil Tay Rushed To Hospital After Being Beat By Group Of Children At A Playground',
        'Post Malone’s Tour Manager Quits Says Post Malone Smells Like Expired Milk And Moldy Cheese',
        'Putin: Clinton Illegally Accepted $400 Million From Russia During Election',
        'Elon Musk: 99.9% Of Media Is Owned By The ’New World Order’'
    ]

    legit_news = [
        'Scientists Develop New Method to Create Stem Cells Without Killing Human Embryos',
        'Luis Palau Diagnosed With Stage 4 Lung Cancer',
        '1st black woman nominated to be Marine brigadier general',
        'Disney CEO Bob Iger revealed that he seriously explored running for president',
        'Trump Has Canceled Via Twitter His G20 Meeting With Vladimir Putin',
        'US Mexico and Canada sign new USMCA trade deal',
        'Afghanistan Women children among 23 killed in US attack UN',
        'UNESCO adds reggae music to global cultural heritage list',
        'The Saudi women detained for demanding basic human rights',
        'Georgia ruling party candidate Zurabishvili wins presidential runoff'
    ]

    scraped_fake_results = {}
    for index, news in enumerate(fake_news):
        scraped_fake_results[index] = {}

        for lang in TARGET_LANGS:
            scraped_fake_results[index][lang] = []

            try:
                translated_news = {'headline': translator.translate(news, dest=lang)}
            except:
                print('Translation issue.')
                translated_news = {'headline': news}

            request = generate_request_cleaned(translated_news)
            search_results = extract_search_results(request, date_from='1/1/2018', date_to='1/1/2019')

            for search_result in search_results:
                scraped_fake_results[index][lang].append(search_result)

            # saving scraped multilingual evidence
            with open(save_fake_to + '.pkl', 'wb') as file:
                pickle.dump(scraped_fake_results, file, pickle.HIGHEST_PROTOCOL)

    scraped_legit_results = {}
    for index, news in enumerate(legit_news):
        scraped_legit_results[index] = {}

        for lang in TARGET_LANGS:
            scraped_legit_results[index][lang] = []

            try:
                translated_news = {'headline': translator.translate(news, dest=lang)}
            except:
                print('Translation issue.')
                translated_news = {'headline': news}

            request = generate_request_cleaned(translated_news)
            search_results = extract_search_results(request, date_from='1/1/2018', date_to='1/1/2019')

            for search_result in search_results:
                scraped_legit_results[index][lang].append(search_result)

            # saving scraped multilingual evidence
            with open(save_legit_to + '.pkl', 'wb') as file:
                pickle.dump(scraped_legit_results, file, pickle.HIGHEST_PROTOCOL)
                
                
def covid_news_scraping(dataset_path, save_to):
    dataset = pd.read_csv(dataset_path)
    
    scraped_results = {}
    
    if os.path.exists(save_to + '.pkl'):
        print('Loading already scraped.')
        with open(save_to + '.pkl', 'rb') as file:
            scraped_results = pickle.load(file)
            
        scraped_df = dataset[dataset.news_id.isin(list(scraped_results.keys()))]
        dataset = dataset[~dataset.index.isin(scraped_df.index)]
    
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        scraped_results[row['news_id']] = {}
        
        for lang in TARGET_LANGS:
            scraped_results[row['news_id']][lang] = []
            
            try:
                translated_news = {'headline': translator.translate(row['title'], dest=lang)}
            except:
                print('Translation issue.')
                translated_news = {'headline': row['title']}

            request = generate_request_cleaned(translated_news)
            search_results = extract_search_results(request)

            for search_result in search_results:
                scraped_results[row['news_id']][lang].append(search_result)
            
            
            # saving scraped multilingual evidence
            with open(save_to + '.pkl', 'wb') as file:
                pickle.dump(scraped_results, file, pickle.HIGHEST_PROTOCOL)