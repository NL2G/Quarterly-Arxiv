import collections

import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('outs/eval.out', 'r') as f:
    data = f.read().strip().split('='*20)
print(len(data))


allscores = []
allscores_intro = []
for i, d in enumerate(data):
    ss = d.split('='*10)
    # title ss[0]
    if len(ss) >= 2:
        abstract = ss[1].replace('\n\n', '\n')
        #print(len(abstract.split('\n')))
        scores = []
        for line in abstract.split('\n'):
            score = line.split(':')[-1].replace(' ', '').replace('[', '').replace(']', '')
            if score.isdigit():
                scores.append(int(score))
        assert len(scores) == 5
        allscores.append(scores)
        if len(ss) >= 3 and i <= 39:
            abstract = ss[2].replace('\n\n', '\n')
            #print(len(abstract.split('\n')))
            scores = []
            for line in abstract.split('\n'):
                score = line.split(':')[-1].replace(' ', '').replace('[', '').replace(']', '')
                if score.isdigit():
                    scores.append(int(score))
            assert len(scores) == 5, d
            allscores_intro.append(scores)
    #intro = ss[2]
allscores = np.array(allscores)
#allscores_intro = np.array(allscores_intro)
print(allscores.shape)
#print(allscores_intro)

with open('outs/eval_sample_intro.out', 'r') as f:
    sample_intro = f.read().strip().split('='*20)

for d in sample_intro:
    d = d.replace('\n\n', '\n')
    scores = []
    try:
        for line in d.split('\n'):
            score = line.split(':')[-1].replace(' ', '').replace('[', '').replace(']', '')
            if score.isdigit():
                scores.append(int(score))
        assert len(scores) == 5, d
        allscores_intro.append(scores)
    except:
        pass


allscores_intro = np.array(allscores_intro)
print(allscores_intro.shape)

cridict = {
    1: 'Fluency',
    2: 'Clarity',
    3: 'Grammaticality',
    4: 'Readability',
    5: 'Coherence'
}
r = collections.defaultdict(list)
for i in range(5):
    c = cridict[i+1]
    # abstract
    abstop = allscores[:40, i]
    absrandom = allscores[40:, i]

    # intro
    introtop = allscores_intro[:27, i]
    introrandom = allscores_intro[27:, i]

    r['List'].append('Top40')
    r['Criterion'].append(c)
    r['Content'].append('Abstract')
    r['Score'].append(np.mean(abstop))

    r['List'].append('Random')
    r['Criterion'].append(c)
    r['Content'].append('Abstract')
    r['Score'].append(np.mean(absrandom))

    r['List'].append('Top40')
    r['Criterion'].append(c)
    r['Content'].append('Introduction')
    r['Score'].append(np.mean(introtop))

    r['List'].append('Random')
    r['Criterion'].append(c)
    r['Content'].append('Introduction')
    r['Score'].append(np.mean(introrandom))

r = pd.DataFrame(r)
print(r)

plt.figure(figsize=(4, 4))
ax = sns.barplot(data=r[r.Content == 'Abstract'], x='Criterion', y='Score', hue='List')
plt.xticks(rotation=15)
plt.ylim((0,5))
ax.get_legend().set_visible(False)
plt.tight_layout()
#plt.savefig('plots/analysis/abs_eval.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

plt.figure(figsize=(4, 4))
ax = sns.barplot(data=r[r.Content == 'Introduction'], x='Criterion', y='Score', hue='List')
plt.xticks(rotation=15)
plt.ylim((0,5))
#ax.get_legend().set_visible(False)
plt.tight_layout()
#plt.savefig('plots/analysis/intro_eval.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
#raise ValueError


# smart
df = pd.read_csv('outs/top40_smart.tsv', sep='\t')
dfsample = pd.read_csv('outs/sample100_smart.tsv', sep='\t')
print(df)
print(dfsample)


# abstract
def get_answer(text):
    answers = []

    text = text.replace(':\n', ':')
    lines = text.strip().split('\n') #if ':\n' not in text else text.strip().split(':\n')
    for line in lines:
        if line.startswith('Evidence_'):
            line = line.replace("[", "").replace("]", "")
            ans = line.split(':')[1].split(';')[0].strip()
            #assert ans.lower() in ['yes', 'no', 'partial'], f"{text}\n{line}\n{ans}"
            if ans.lower() == 'yes':
                answers.append(1)
            else:
                answers.append(0)
                #print("line: ", line)
            #answers.append(ans)
    assert len(answers) == 5, f"{text}\n{answers}"
    return answers


# abstract
answers_top = np.array([get_answer(text) for text in df['abstract_smart']])
answers_random = np.array([get_answer(text) for text in dfsample['abstract_smart']])

#
answers_top_intro = np.array([get_answer(text) for text in df['introduction_smart'] if isinstance(text, str)])
answers_random_intro = np.array([get_answer(text) for text in dfsample['introduction_smart']])

#print(answers_top_intro.shape)
#print(answers_random_intro.shape)


cridict = {
    1: 'Specific',
    2: 'Measurable',
    3: 'Achievable',
    4: 'Relevant',
    5: 'Time-bound'
}

#data = pd.DataFrame({'List': ['Top40', 'Sample'] * 2, 'content': ['Abstract', 'Introduction']*2, 'Percent': [answers_top]})
r = collections.defaultdict(list)
for i, criterion in cridict.items():
    if i in [3, 5]:
        continue
    abs_top = answers_top[:, i-1]
    r['List'].append('Top40')
    r['Content'].append('Abstract')
    r['Percent'].append(sum(abs_top)/len(abs_top) * 100)
    r['Criterion'].append(criterion)

    abs_random = answers_random[:, i-1]
    r['List'].append('Random')
    r['Content'].append('Abstract')
    r['Percent'].append(sum(abs_random) / len(abs_random) * 100)
    r['Criterion'].append(criterion)

    intr_top = answers_top_intro[:, i-1]
    #print(intr_top)
    r['List'].append('Top40')
    r['Content'].append('Introduction')
    r['Percent'].append(sum(intr_top) / len(intr_top) * 100)
    r['Criterion'].append(criterion)

    intr_random = answers_random_intro[:, i-1]
    r['List'].append('Random')
    r['Content'].append('Introduction')
    r['Percent'].append(sum(intr_random) / len(intr_random) * 100)
    r['Criterion'].append(criterion)

#print([len(v) for k, v in r.items()])
r = pd.DataFrame(r)
#print(r)

plt.figure(figsize=(4,4))
ax=sns.barplot(data=r[r.Content=='Abstract'], x='Criterion', y='Percent', hue='List')
ax.get_legend().set_visible(False)
plt.tight_layout()
plt.savefig('plots/analysis/abs_smart.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(4,4))
sns.barplot(data=r[r.Content=='Introduction'], x='Criterion', y='Percent', hue='List')
plt.tight_layout()
plt.savefig('plots/analysis/intro_smart.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# 3 moves
with open('outs/3moves_new.out', 'r') as f:
    data = f.read().strip().split('='*20)

print(len(data))

print(data[-1])

abst, intro = [], []
for i, d in enumerate(data[:-1]):
    ss = d.split('='*10)
    if len(ss) >= 2:
        for line in ss[1].split('\n'):
            if line.startswith('3 moves:'):
                if 'yes' in line.lower():
                    abst.append(1)
                else:
                    abst.append(0)
        if len(ss) >= 3:
            for line in ss[2].split('\n'):
                if line.startswith('3 moves:'):
                    if 'yes' in line.lower():
                        intro.append(1)
                    else:
                        intro.append(0)
print(len(abst))
print(len(intro))

r = collections.defaultdict(list)

r['Content'].append('Abstract')
r['List'].append('Top40')
r['Percent'].append(sum(abst[:40])/40 * 100)

r['Content'].append('Abstract')
r['List'].append('Random')
r['Percent'].append(sum(abst[40:])/100 * 100)

r['Content'].append('Introduction')
r['List'].append('Top40')
r['Percent'].append(sum(intro[:27])/27 * 100)

r['Content'].append('Introduction')
r['List'].append('Random')
r['Percent'].append(sum(intro[27:])/99 * 100)

r = pd.DataFrame(r)

plt.figure(figsize=(4,3))
ax = sns.barplot(data=r, x='Content', y='Percent', hue='List')
ax.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('plots/analysis/3moves.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
raise ValueError


# title type
with open('outs/title.out', 'r') as f:
    data = f.read().strip().split('='*20)[:-1]


typs = []
for d in data:
    for line in d.split('\n'):
        if line.startswith("Type: "):
            typ = line.split('Type: ')[-1]
            typs.append('Type '+typ)

assert len(typs) == 140

top_type = collections.Counter(typs[:40])
random_type = collections.Counter(typs[40:])

for k, v in random_type.items():
    if k not in top_type.keys():
        top_type[k] = 0

for k, v in top_type.items():
    if k not in random_type.keys():
        random_type[k] = 0

top_type = dict(sorted(top_type.items(), key=lambda x: x[0]))
random_type = dict(sorted(random_type.items(), key=lambda x: x[0]))
print(top_type)
print(random_type)
#raise ValueError

#plt.figure(figsize=(3.5,3.5))
fig1, ax1 = plt.subplots(figsize=(3.5,3.5))
#ax1.pie(list(top_type.values()), labels=list(top_type.keys()), autopct='%1.1f%%', pctdistance=0.7, startangle=0)
ax1.pie(list(top_type.values()), autopct='%1.1f%%', pctdistance=1.1, startangle=0)
#plt.legend(ax1.patches, list(top_type.keys()), ncol=1, bbox_to_anchor=(1,.7))
plt.tight_layout()
plt.savefig('plots/analysis/top_title.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


#plt.figure(figsize=(3.5,3.5))
fig1, ax1 = plt.subplots(figsize=(3.5,3.5))
#ax1.pie(list(top_type.values()), labels=list(top_type.keys()), autopct='%1.1f%%', pctdistance=0.7, startangle=0)
ax1.pie(list(random_type.values()), autopct='%1.0f%%', pctdistance=1.1, startangle=0)
#plt.legend(ax1.patches, list(top_type.keys()), ncol=1, bbox_to_anchor=(1,.7))
plt.tight_layout()
plt.savefig('plots/analysis/random_title.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.legend(ax1.patches, list(top_type.keys()), ncol=1, bbox_to_anchor=(1,.7))
plt.tight_layout()
plt.savefig('plots/analysis/title_legend.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
raise ValueError

# number of figures/tables
dftop = pd.read_csv("data/arxiv/analyzed/arxiv.xlsx - top40.csv")
dftop['List'] = ['Top40'] * len(dftop)
dfrandom = pd.read_csv("data/arxiv/analyzed/arxiv.xlsx - sampled100.csv")
dfrandom['List'] = ['Random'] * len(dfrandom)
df = pd.concat([dftop, dfrandom], ignore_index=True)
df['num_figures'] = df.apply(lambda x: x['num_figures'] / x['num_pages'], axis=1)
df['num_tables'] = df.apply(lambda x: x['num_tables'] / x['num_pages'], axis=1)
#print(df)

cp = sns.color_palette() 
#print(cp)
#raise ValueError
plt.figure(figsize=(4,3))
ax = sns.histplot(data=df[df.List=='Top40'], x='num_tables', stat='percent', binwidth=0.1, label='Top40')
sns.histplot(data=df[df.List=='Random'], x='num_tables', stat='percent', binwidth=0.1, label='Random', ax=ax)
plt.axvline(np.mean(df[df.List=='Top40']['num_tables']), color='red', linestyle='--', label='Avg Top40')
plt.axvline(np.mean(df[df.List=='Random']['num_tables']), color='red', label='Avg Random')
plt.xlim(0, max(df['num_tables']))
#lgd, keys = ax.get_legend_handles_labels()
#d = dict(zip(keys, lgd))
#plt.legend(d.values(), d.keys())
plt.xlabel('Normalized Number of Tables')

plt.tight_layout()
#plt.savefig('plots/analysis/num_tables.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

plt.figure(figsize=(4,3))
ax = sns.histplot(data=df[df.List=='Top40'], x='num_figures', stat='percent', binwidth=0.1, label='Top40')
sns.histplot(data=df[df.List=='Random'], x='num_figures', stat='percent', binwidth=0.1, label='Random', ax=ax)
plt.xlim(0, max(df['num_figures']))
plt.axvline(np.mean(df[df.List=='Top40']['num_figures']), color='red', linestyle='--', label='Avg Top40')
plt.axvline(np.mean(df[df.List=='Random']['num_figures']), color='red', label='Avg Random')
lgd, keys = ax.get_legend_handles_labels()
d = dict(zip(keys, lgd))
plt.legend(d.values(), d.keys())
plt.xlabel('Normalized Number of Figures')

plt.tight_layout()
#plt.savefig('plots/analysis/num_figs.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()





