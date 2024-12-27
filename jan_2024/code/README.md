
- ``crawl_arxiv.py`` and ``normalize_citation_count.py`` are used for crawling arxiv papers with the Samantic Scholar API and normalizing the citation counts; the output file [arxiv_20230101-20240131_20240221_71139_zscore_sorted.csv](../data/arxiv_20230101-20240131_20240221_71139_zscore_sorted.csv) is in the data folder.


- ``inspect_comments.py`` is used to find the most contributing n-grams in comments.


- Code for prompting GPT4o: ``gpt4call_3moves.py``, ``gpt4call_smart.py``, ``gpt4call_title.py`` and ``gpt4call_eval.py``. The outputs are sotred in folder [data/outs](../data/outs/).


- Find code for plotting:
    - ``plot_arxiv_statistics.py`` and ``find_topics.py``: plots in Section **Data Overview**
    - ``analyze_top40.py``: plots in Section **Top-N papers**
    - ``plot_eval.py``: plots in Section **Writing analysis**

- The plots are stored in folder [plots](../plots/)








