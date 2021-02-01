import os

DATA_PATH = '../data/fakeNewsDatasets/'
NEWS_FAKE = os.path.join(DATA_PATH, 'fakenewsdataset/fake/')
NEWS_LEGIT = os.path.join(DATA_PATH, 'fakenewsdataset/legit/')
CELEB_FAKE = os.path.join(DATA_PATH, 'celebritydataset/fake/')
CELEB_LEGIT = os.path.join(DATA_PATH, 'celebritydataset/legit/')

MULT_EVIDENCE_PATH = os.path.join(DATA_PATH, 'multilingual_evidence_1/')
MULT_EVIDENCE_NEWS_FAKE = os.path.join(MULT_EVIDENCE_PATH, 'fakenewsdataset_fake.pkl')
MULT_EVIDENCE_NEWS_LEGIT = os.path.join(MULT_EVIDENCE_PATH, 'fakenewsdataset_legit.pkl')
MULT_EVIDENCE_CELEB_FAKE = os.path.join(MULT_EVIDENCE_PATH, 'celebritydataset_fake.pkl')
MULT_EVIDENCE_CELEB_LEGIT = os.path.join(MULT_EVIDENCE_PATH, 'celebritydataset_legit.pkl')