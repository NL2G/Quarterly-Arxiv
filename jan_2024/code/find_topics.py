import string
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np

def generate_ngram(text, ngram=1):
    text = ''.join([c if c not in string.punctuation else " " for c in text]).lower()
    tokenzied = text.split()
    stopwlist = stopwords.words('english') + ['experiment', 'experiments', 'github',
                                              'show', 'demonstrate', 'experimental', 'results',
                                              'result', 'demonstrates', 'shows', 'http', 'https']
    ngrams = []
    for i in range(len(tokenzied)):
        if i+ngram <= len(tokenzied):
            ng = tokenzied[i:i+ngram]
            contain_sw = False
            for sw in stopwlist:
                if sw in ng:
                    contain_sw = True
                    break
            if not contain_sw:
                ngrams.append('-'.join(ng))
    ngrams = list(set(ngrams))
    return ngrams


def get_topk_ngram(texts, ngram=1, k=10):
    ngrams = [t for text in texts for t in generate_ngram(text, ngram=ngram)]
    c = Counter(ngrams).most_common(k)
    return c


def contain_keywords(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


if __name__ == '__main__':
    df = pd.read_csv('data/arxiv/arxiv_20240221_with_std_sorted.csv', sep='\t')
    dfall = pd.read_csv('data/arxiv/arxiv_20230101-20240131_20240221_71139.csv', sep='\t')
    df['summary'] = [dfall[dfall.entry_id==i]['summary'].values[0] for i in df['entry_id']]
    #df.sort_values('week', ascending=True, inplace=True)
    df['Week'] = df['week'].apply(lambda x: x.split('/')[0])
    df['Month'] = df['published'].map(lambda x: x[:7])

    '''
    for week, group in df.groupby('Month'):
        print()
        print(week)
        #topics = get_topk_ngram(list(group['summary']), ngram=3, k=10)
        topics = get_topk_ngram(list(group['title']), ngram=3, k=10)
        print(topics)
        for topic, count in topics:
            print(f"{topic}-{round(count/len(group)*100, 1)}")
    '''
    keywords = [['llm', 'llms', 'large language model', 'large language models'],
                ['chatgpt', 'chat-gpt', 'gpt-4', 'gpt4', 'gpt 4', 'gpt 3.5', 'gpt3.5', 'gpt-3.5'],
                ['llama']]
    keyd = [
        'LLM',
        'GPT-series',
        'LLaMA'
    ]

    allkeywords = [tt for t in keywords for tt in t]
    r = defaultdict(list)
    r2 = defaultdict(list)
    for week, group in df.groupby('Week'):
        texts = [a + " " + t for a, t in zip(group['summary'], group['title'])]

        perc = np.sum([contain_keywords(t, allkeywords) for t in texts]) / len(group) * 100
        r['Keywords'].append('Any')
        r['Week'].append(week)
        r['Percentage (%)'].append(perc)


        for i, keyword in enumerate(keywords):
            perc = np.sum([contain_keywords(t, keyword) for t in texts]) / len(group) * 100
            r['Keywords'].append(keyd[i])
            r['Week'].append(week)
            r['Percentage (%)'].append(perc)


    r = pd.DataFrame(r)

    plt.figure(figsize=(9, 3))
    sns.set_style('darkgrid')
    sns.lineplot(data=r, x='Week', y='Percentage (%)', hue='Keywords')
    plt.xticks(rotation=90)
    plt.xlim(-1, len(set(r['Week'])))
    plt.tight_layout()
    plt.savefig('plots/llm_popularity.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    #print(topics)

    #print([t for t in df['title'] if 'multi agent' in t.lower() and 'reinforcement' in t.lower()])
    #print([t for t in df['title'] if 'multi agent' in t.lower()])


