import argparse
from tools.web_scraping import multilingual_evidence_scraping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", help="Path to the datasets from FakeNewsDatasets.")
    parser.add_argument("--path", help="Where to save scraped evidence.")
    
    args = parser.parse_args()
    
    if args.data and args.path:
        multilingual_evidence_scraping(args.data, args.path)