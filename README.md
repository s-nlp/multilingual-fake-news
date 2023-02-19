# Multiverse: Multilingual Evidence for Fake News Detection

The code for the computation of **Multiverse** feature. The feature was firstly [presented](https://ieeexplore.ieee.org/abstract/document/9260100) shortly at DSAA conference, following with extended experiments in ACL SRW 2021 [paper](https://aclanthology.org/2021.acl-srw.32/). The full version of experiments with more detailed feature description can be found at [arXiv]().

## Approach

![](https://github.com/s-nlp/multilingual-fake-news/blob/master/main_schema.jpg)

**Step 1. Text extraction:** As a new article arrives, the title and content are extracted from it.

**Step 2. Text translation:** The title is translated into target languages and new search requests are generated. 

**Step 3. Cross-lingual news retrieval:** Based on generated cross-lingual request -- translated title -- the search with a Web search engine is executed.

**Step 4. Cross-lingual evidence impact computation:** Top-N articles from search results are extracted to assess the authenticity of the initial news. The information described in the news is compared with the information in the articles from the search result. Also, the ranks of the source of the extracted articles are taken into account. The number of articles that confirms or disproves the original news from reliable sources is estimated.

**Step 5. News classification:** Based on the information from the previous step, the decision is made about the authenticity of the news. If the majority of results support the original news, then it is more likely to be true; if there are contradictions -- it is a signal to consider the news as a fake.

*We performed a study where Step 4 and 5 were conducted not automatically but by human annotators. Its annotation schema is avaialble in [this spreadsheet](https://docs.google.com/spreadsheets/d/16HSCTSJhjdPNAD_IEHKBJS1fL_j6PPl0dCqUGlo6t3Q/edit?usp=sharing). It may be useful for reproducing the original study described in the paper below.*

## Code

`evidence_scraping.py`: the main entrance to the code, that runs Multiverse feature extraction.

`tools/web_scraping.py`: the main Multiverse web extraction and computation.

`tools/features_extraction.py`: construction of linguistic and Mutliverse features.

`tools/preprocessing.py`: addition texts preprocessing.

## Citation

```
@inproceedings{dementieva-panchenko-2021-cross,
    title = "Cross-lingual Evidence Improves Monolingual Fake News Detection",
    author = "Dementieva, Daryna  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-srw.32",
    doi = "10.18653/v1/2021.acl-srw.32",
    pages = "310--320",
    abstract = "Misleading information spreads on the Internet at an incredible speed, which can lead to irreparable consequences in some cases. Therefore, it is becoming essential to develop fake news detection technologies. While substantial work has been done in this direction, one of the limitations of the current approaches is that these models are focused only on one language and do not use multilingual information. In this work, we propose a new technique based on cross-lingual evidence (CE) that can be used for fake news detection and improve existing approaches. The hypothesis of the usage of cross-lingual evidence as a feature for fake news detection is confirmed, firstly, by manual experiment based on a set of known true and fake news. Besides, we compared our fake news classification system based on the proposed feature with several strong baselines on two multi-domain datasets of general-topic news and one newly fake COVID-19 news dataset showing that combining cross-lingual evidence with strong baselines such as RoBERTa yields significant improvements in fake news detection.",
}
```

## Contacts

If you find some issue, do not hesitate to add it to [Github Issues](https://github.com/s-nlp/multilingual-fake-news/issues).

For any questions, please contact: Daryna Dementieva (daryna.dementieva@tum.de)
