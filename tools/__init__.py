from google import google
from contextlib import contextmanager
import re
import signal
import requests
from bs4 import BeautifulSoup
from scipy.spatial import distance
from settings.diffbot import client
from settings.scraper import *
from settings.similarity import bert_embedding


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


def generate_request(news):
    return news['headline']


def calculate_sent_embd(sentence):
    if len(sentence) == 0:
        return 0
    embeddings = bert_embedding([sentence.lower()], 'avg')[0][1]
    return sum(embeddings) / len(sentence)


def extract_search_results(request):
    search_results = google.search(request, NUM_RESULT_PAGES)

    result_news = []
    for search_result in search_results:
        search_news_headline = ""
        search_news_content = ""

        search_link = search_result.link
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
            ############################

            result_news.append({'url': search_link,
                                'headline': search_news_headline.strip(),
                                'content': search_news_content.strip()})

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