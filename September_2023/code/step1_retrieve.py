#!/usr/bin/env python
import os
import types
from collections import defaultdict

import arxiv
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from semanticscholar.Paper import Paper
from tqdm import tqdm
from pandas import DataFrame
from datetime import datetime
from itertools import chain
from os.path import basename
from semanticscholar import SemanticScholar
import os
os.chdir("/Users/zhangran/Documents/GitHub/Quaterly-Arxiv/code")
def search(queries=[], field="all", cats=["cs.AI", "cs.CV"], sort_order = ""):  # cs.AI, cs.CV "cs.CL", "cs.LG"
    # Use the arxiv API to query for papers from specified categories
    query_string, client = "", arxiv.Client(num_retries=40, page_size=1000)
    if queries:
        query_string += "(" + " OR ".join(f"{field}:{query}" for query in queries) + ")"
    if cats:
        if query_string:
            query_string += " AND "
        query_string += "(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")"
    print(query_string)
    return client.results(arxiv.Search(
        query=query_string,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

def _get_papers(
        # require version 0.4.0
        self,
        paper_ids,
        fields: list = None
):
    # Overwriting this method of the python semanticscholar API and allow papers that exist in arxiv but not semantic scholar
    if not fields:
        fields = Paper.SEARCH_FIELDS

    url = f'{self.api_url}/paper/batch'

    fields = ','.join(fields)
    parameters = f'&fields={fields}'

    payload = {"ids": paper_ids}

    data = self._requester.get_data(
        url, parameters, self.auth_header, payload)
    papers = [Paper(item) if item else None for item in data]

    return papers

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


sch = SemanticScholar(timeout=120)
# Overwrite function with hotfix
sch.get_papers = types.MethodType(_get_papers, sch)

papers = defaultdict(list)
results = []
batch = []
progress = 0
print("Progress:", 0)
cats=["cs.CL", "cs.LG"] # , "cs.AI", "cs.LG", "cs.CL"， “cs.CV”
if "results.csv" in os.listdir("../data/"): # to prevent double downloading, load the collected data, depending on the stage, this file may differ
    df_ = pd.read_csv("../data/results.csv", sep="\t")
    print("Collected: ", len(df_))
    entry_list = df_["entry_id"].tolist()
else:
    df = pd.DataFrame()
    entry_list = []
for result in search(cats = cats):
    if (result.published.year >= 2023) & (not result.entry_id in entry_list):
        results.append(result)
        batch.append("arxiv:" + basename(result.entry_id).split("v")[0])
        progress += 1
        if len(batch)%500 == 0:
            citation = get_citations(batch)
            for r, cnt in zip(results, citation):
                r.citationCount = cnt
                papers[r.published.month].append(r)
                results = []
                batch = []
            print("Progress:", progress)
            df = pd.DataFrame([vars(a) for a in sum(dict(papers).values(), [])])
            batch = []
            df.to_csv("../data/progress.csv", sep="\t", index = False)
    elif result.published.year < 2023:
        print("Finished!!!!!!")
        break


if len(batch)>0:
    citation = get_citations(batch)
    for r, cnt in zip(results, citation):
        r.citationCount = cnt
        papers[r.published.month].append(r)
df = pd.DataFrame([vars(a) for a in sum(dict(papers).values(), [])])
if len(df_) > 0:
    df = pd.concat([df_, df])
df.to_csv("../data/results.csv",  sep="\t")