#!/usr/bin/env python
import collections
import types
from collections import defaultdict
from statistics import mean
import shelve
import json
import arxiv
import pandas as pd
import time
import seaborn as sb
from semanticscholar.Paper import Paper
from tqdm import tqdm
from pandas import DataFrame
from datetime import datetime
from itertools import chain
from os.path import basename
from semanticscholar import SemanticScholar
import numpy as np

def search(queries=[], field="all", cats=["cs.CL", "cs.LG", 'cs.AI', 'cs.CV'], start='202301010000', end='2023070100'):  # cs.AI, cs.CV
    # Use the arxiv API to query for papers from specified categories
    query_string, client = "", arxiv.Client(num_retries=40, page_size=1000)
    if queries:
        query_string += "(" + " OR ".join(f"{field}:{query}" for query in queries) + ")"
    if cats:
        if query_string:
            query_string += " AND "
        query_string += "(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")"
    #query_string += " AND submittedDate:[202301010000 TO 2023070100]"
    query_string += f" AND submittedDate:[{start} TO {end}]"
    print(query_string)
    #raise ValueError
    return client.results(arxiv.Search(
        query=query_string,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        #max_results=500
    ))


def _get_papers(
        self,
        paper_ids,
        fields: list = None
):
    # Overwriting this method of the python semanticscholar API and allow papers that exist in arxiv but not semantic scholar
    if not fields:
        fields = Paper.SEARCH_FIELDS

    #url = f'{self._AsyncSemanticScholar.api_url}/paper/batch'
    # new api url
    url = f'{self._AsyncSemanticScholar.api_url}/graph/v1/paper/batch'
    fields = ','.join(fields)
    parameters = f'&fields={fields}'

    payload = {"ids": paper_ids}
    import asyncio

    while (True):
        try:
            data = asyncio.run(self._AsyncSemanticScholar._requester.get_data_async(
                url, parameters, self._AsyncSemanticScholar.auth_header, payload))
            break
        except:
            print('time out; sleep 2 seconds...')
            time.sleep(2)
            continue
    papers = [Paper(item) if item else None for item in data]
    return papers


def get_papers(file="papers.shelf", cached=False, start='202301010000', end='202307010000'):
    # Get raw data from arxiv and semantic scholar and save it to papers.shelf
    papers = defaultdict(list)

    if cached:
        with shelve.open(file, "r") as shelf:
            print("Loading cached papers.")
            for month in shelf:
                papers[int(month)] = shelf[month]
    else:
        print("Downloading papers.")
        results = []
        batch = []
        progress = 0
        print("Progress:", 0)
        count = 1
        for result in search(start=start, end=end):
            if len(batch) < 500:
                results.append(result)
                batch.append("arxiv:" + basename(result.entry_id).split("v")[0])
                progress += 1
            else:
                for r, cnt in zip(results, get_citations(batch)):
                    r.citationCount = cnt
                    papers[f"{r.published.year}-{r.published.month}"].append(r)
                    results = []
                    batch = []

                print(f"{result.published.year}-{result.published.month}-Progress{count}:", progress)
                count += 1
                #df = pd.DataFrame([vars(a) for a in sum(dict(papers).values(), [])])
                #df.to_csv(f"data/arxiv/arxiv_{args.year}_6.csv", index=False, sep='\t')

        if len(results) > 0:
            for r, cnt in zip(results, get_citations(batch)):
                r.citationCount = cnt
                papers[f"{r.published.year}-{r.published.month}"].append(r)
            print("Progress:", progress)

    return papers


sch = SemanticScholar(timeout=100)
# Overwrite function with hotfix
sch.get_papers = types.MethodType(_get_papers, sch)


def get_citations(batch):
    # Get citation counts with overwritten semantic scholar batch api
    papers = sch.get_papers(paper_ids=batch)
    citation_counts = []
    for p in papers:
        try:
            citation_counts.append(p.citationCount)
        except:
            citation_counts.append(0)
    return citation_counts


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--year", default=2024, type=int)
    parser.add_argument("--start", default='202301010000', type=str)
    parser.add_argument("--end", default='202307010000', type=str)
    parser.add_argument('--use_cache', action='store_true')

    args = parser.parse_args()
    print(args)

    sb.set()
    
    papers = get_papers(cached=False, start=args.start, end=args.end)
    df = pd.DataFrame([vars(a) for a in sum(dict(papers).values(), [])])
    df.to_csv(f"data/arxiv/arxiv_{args.start}_{args.end}.csv", index=False, sep='\t')
