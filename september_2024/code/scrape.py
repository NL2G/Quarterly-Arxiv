#!/usr/bin/env python
import types
import shelve
import arxiv
import pandas as pd
import time
from collections import defaultdict
from os.path import basename, exists
from semanticscholar.Paper import Paper
from semanticscholar import SemanticScholar
import logging
logging.debug = print
logging.info = print

FIELDS = [
    'abstract',
    'authors',
    'authors.affiliations',
    'authors.externalIds',
    'authors.url',
    'authors.homepage',
    'citationCount',
    'citationStyles',
    'corpusId',
    'externalIds',
    'fieldsOfStudy',
    'influentialCitationCount',
    'isOpenAccess',
    'journal',
    'openAccessPdf',
    'paperId',
    'publicationDate',
    'publicationTypes',
    'publicationVenue',
    'referenceCount',
    's2FieldsOfStudy',
    'title',
    'url',
    'venue',
    'year'
]


def search(queries=None, field="all", cats=["cs.CL", "cs.LG", 'cs.AI', 'cs.CV'], start='202408120000', end='2024093000'):
    query_string, client = "", arxiv.Client(num_retries=2000, page_size=2000, delay_seconds=10)
    if queries:
        query_string += "(" + " OR ".join(f"{field}:{query}" for query in queries) + ")"
    if cats:
        if query_string:
            query_string += " AND "
        query_string += "(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")"
    query_string += f" AND submittedDate:[{start} TO {end}]"
    print(query_string)

    return client.results(arxiv.Search(
            query=query_string,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        ))


def _get_papers(self, paper_ids, fields: list = None):
    if not fields:
        fields = Paper.SEARCH_FIELDS
    url = f'{self._AsyncSemanticScholar.api_url}/graph/v1/paper/batch'
    fields = ','.join(fields)
    parameters = f'&fields={fields}'
    payload = {"ids": paper_ids}
    import asyncio

    while True:
        try:
            data = asyncio.run(self._AsyncSemanticScholar._requester.get_data_async(
                url, parameters, self._AsyncSemanticScholar.auth_header, payload))
            break
        except Exception as e:
            print('time out; sleep 2 seconds...', e)
            time.sleep(4)
            continue
    papers = [Paper(item) if item else None for item in data]
    return papers


def get_papers(file="papers.shelf", start='202401010000', end='202307010000', backup_interval=10000):
    papers = defaultdict(list)

    results, batch = [], []
    progress, count = 0, 1
    backup_counter = 0

    with shelve.open(file, "c") as shelf:
        for result in search(start=start, end=end):
            if result is None:
                continue
            if len(batch) < 500:
                results.append(result)
                batch.append("arxiv:" + basename(result.entry_id).split("v")[0])
                progress += 1
                backup_counter += 1
            else:
                for r, raw_scholar in zip(results, get_citations(batch)):
                    r.raw_scholar_data = raw_scholar
                    if not r in papers[f"{r.published.year}-{r.published.month}"]:
                        papers[f"{r.published.year}-{r.published.month}"].append(r)
                results, batch = [], []

                if backup_counter >= backup_interval:
                    shelf.clear()
                    shelf.update(papers)
                    backup_counter = 0
                    print(f"Backup saved at interval {count}.")

                print(f"{result.published.year}-{result.published.month}-Progress {count}:", progress)
                count += 1

        if len(results) > 0:
            for r, raw_scholar in zip(results, get_citations(batch)):
                r.raw_scholar_data = raw_scholar
                papers[f"{r.published.year}-{r.published.month}"].append(r)
            print("Progress:", progress)

    return papers



sch = SemanticScholar(timeout=100)
sch.get_papers = types.MethodType(_get_papers, sch)


def get_citations(batch):
    papers = sch.get_papers(fields=FIELDS, paper_ids=batch)
    raw_data = []
    for p in papers:
        try:
            raw_data.append(p.raw_data)
        except Exception as e:
            raw_data.append(None)
    return raw_data


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--start", default='202301010000', type=str)
    parser.add_argument("--end", default='202409300000', type=str)

    args = parser.parse_args()
    print(args)

    papers = get_papers(start=args.start, end=args.end)
    df = pd.DataFrame([vars(a) for a in sum(dict(papers).values(), [])])
    df['updated'] = df['updated'].dt.tz_localize(None)
    df['published'] = df['published'].dt.tz_localize(None)

    df.to_json(f"data/arxiv/arxiv_{args.start}_{args.end}.json", index=False)
    df.to_excel(f"data/arxiv/arxiv_{args.start}_{args.end}.xlsx", index=False)
