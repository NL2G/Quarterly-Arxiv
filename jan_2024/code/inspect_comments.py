import collections

import numpy as np

from find_topics import *

df = pd.read_csv('data/arxiv/arxiv_20230101-20240131_20240221_71139.csv', sep='\t')

tmp = df[(~df.comment.isna()) & (df.comment.str.lower().str.contains('neurips'))]
cv = len(tmp[tmp.primary_category=='cs.CV'])
cl = len(tmp[tmp.primary_category=='cs.CL'])
ai = len(tmp[tmp.primary_category=='cs.AI'])
lg = len(tmp[tmp.primary_category=='cs.LG'])
print(cv/len(tmp))
print(cl/len(tmp))
print(ai/len(tmp))
print(lg/len(tmp))
raise ValueError
#df = df[df.primary_category=='cs.CL']

df['published'] = [pd.to_datetime(date_string, errors='coerce') for date_string in df['published']]
df['week'] = df['published'].apply(lambda x: x.to_period('W-SAT').strftime(None))

ngram_std = defaultdict(list)
ngram_perc = defaultdict(dict)
for week, group in df.groupby('week'):
    comments = [str(c) for c in group['comment']]
    ngrams = get_topk_ngram(comments, ngram=2, k=None)
    #ngrams = get_topk_ngram(comments, ngram=1, k=None)
    for w, c in ngrams:
        perc = c/len(comments)
        ngram_std[w].append(perc)
        #ngram_std[w].append(c)
        #ngram_perc[w][week] = perc
        ngram_perc[w][week] = c

for w, p in ngram_std.items():
    diff = len(set(df['week'])) - len(p)
    if diff > 0:
        p += [0] * diff
    std = np.std(p)
    ngram_std[w] = std

important_ngrams = sorted(ngram_std.items(), key=lambda x: x[1], reverse=True)[:20]

print(important_ngrams)
for ng, _ in important_ngrams:
    # sort weeks
    sorted_weeks = sorted(ngram_perc[ng].items(), key=lambda x: x[1], reverse=True)[:10]
    #sorted_weeks = sorted(sorted_weeks, key=lambda x: x[0])
    print()
    print(ng)
    print(sorted_weeks)
#print()

'''
df = df[df.week.str.contains('2023-12-03')]


ngrams = get_topk_ngram([str(c) for c in df['comment']], ngram=1, k=100)
print([(w, round(c/len(df)*100, 1)) for w, c in ngrams if w != 'nan'])


#raise ValueError
for i, ng in enumerate(df['comment']):
    ng = str(ng).lower()
    if 'accept' in ng or 'publish' in ng:
        print(ng)
        print(df.iloc[i]['entry_id'])

#print(ngrams)


'''
