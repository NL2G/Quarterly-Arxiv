import collections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

df = pd.read_csv('data/arxiv/top40-2024_02_21 - top40_weekfixed.tsv', sep='\t')
#print(df)
#raise ValueError
df = df[['ranking', 'title', 'ranking923', 'ranking623', 'primary_category']]

df.astype({'ranking623': 'str', 'ranking923': 'str', 'ranking': 'int16'})
print(df)

# available in previous datasets but appear in the top40 list for the first time
tmp = df[((df.ranking923 == '100') & (df.ranking623 == 'False')) | ((df.ranking923=='100') & (df.ranking623 == '100'))]
tmp.to_csv('results/top40_first_appearance.csv', index=False)
print(tmp)

# How many papers in the top40 list are published before 6/30, 9/30, 1/31
# top40
# before 6/30
print(len(df[(df.ranking623!='False')]))
# before 9/30
print(len(df[df.ranking923!='False']))
# rest
print(len(df[(df.ranking923=='False') & (df.ranking623=='False')]))

# top20
tmp = df[df.ranking <= 20]
# before 6/30
print(len(tmp[(tmp.ranking623!='False')]))
# before 9/30
print(len(tmp[tmp.ranking923!='False']))
# rest
print(len(tmp[(tmp.ranking923=='False') & (tmp.ranking623=='False')]))

#raise ValueError
r = pd.DataFrame({'List': ['top20']*4+ ['top40']*4, 'Count': [0, 14, 18, 20, 0, 21, 34, 40], 'Date': ['23/1/1', '23/6/30', '23/9/30', '24/1/31']*2})

plt.figure(figsize=(3,4))
ax = sns.lineplot(data=r, x='Date', y='Count', hue='List', marker='o')
ax.set_yticks(ticks=range(0, 41, 2), labels=range(0, 41, 2))
ax.set_xlim(-0.1,3.1)
ax.set_ylim(-1,41)
plt.tight_layout()
plt.savefig('plots/top40_count_dis.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# For papers before 6/30, which are continueing to increase/decrease?
tmp = df[df.ranking623!='False']
print(tmp)
tmp = tmp.astype({'ranking623': 'int', 'ranking923': 'int'}, errors="ignore")
#print(tmp.ranking623)
increase = tmp[(tmp.ranking < tmp.ranking923) & (tmp.ranking923 < tmp.ranking623)]
decrease = tmp[(tmp.ranking > tmp.ranking923) & (tmp.ranking923 > tmp.ranking623)]
inde = tmp[(tmp.ranking >= tmp.ranking923) & (tmp.ranking923 <= tmp.ranking623)]
dein = tmp[(tmp.ranking <= tmp.ranking923) & (tmp.ranking923 >= tmp.ranking623)]
nochange = tmp[(tmp.ranking == tmp.ranking923) & (tmp.ranking923 == tmp.ranking623)]
tmp = pd.concat([increase, decrease, dein, inde, nochange], ignore_index=True)
tmp['trend'] = ['increase'] * len(increase) + ['decrease'] * len(decrease) + ['decrease-increase'] * len(dein) + ['increase-decrease'] * len(inde) + ['no change'] * len(nochange)
'''
print(nochange)
print(increase)
print(decrease)
print(inde)
print(dein)
'''
print(tmp)
print(len(tmp))
tmp.to_csv('results/top40_3phases_change.csv', index=False)
#print(tmp.duplicated('title'))


# For papers between 6/30 and 9/30, which are increasing/decreasing?
tmp = df[(df.ranking923 != 'False') & (df.ranking623 == 'False')]
tmp = tmp.astype({'ranking923': 'int'})
increase = tmp[tmp.ranking923 > tmp.ranking]
decrease = tmp[tmp.ranking923 < tmp.ranking]
nochange = tmp[tmp.ranking923 == tmp.ranking]
tmp = pd.concat([increase, decrease, nochange], ignore_index=True)
tmp['trend'] = ['increasing'] * len(increase) + ['decreasing'] * len(decrease) + ['no change'] * len(nochange)
print(tmp)
print(len(tmp))
tmp.to_csv('results/top40_2phases_change.csv', index=False)


# distribution of arXiv categories comparison top 40
from collections import Counter
c3 = Counter(df['primary_category'])#.most_common()
c3.update({'cs.CR': 0})
print(c3)

tmp = collections.defaultdict(list)

c1 = {
    "cs.CL": 23, 'cs.CV': 6, 'cs.LG': 6, 'cs.AI': 4, 'cs.RO': 0, 'cs.GR': 0, 'cs.CR': 1
}


for cat, count in c1.items():
    tmp['Data'].append('arxiv-06/23')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 40')

c2 = {
    "cs.CL": 21, 'cs.CV': 10, 'cs.LG': 5, 'cs.AI': 3, 'cs.RO': 1, 'cs.GR': 0, 'cs.CR': 0
}
for cat, count in c2.items():
    tmp['Data'].append('arxiv-09/23')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 40')

for cat, count in c3.items():
    tmp['Data'].append('arxiv-01/24')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 40')

#r = tmp.copy()
#tmp = pd.DataFrame(tmp)
#plt.figure(figsize=(4,3))
#ax = sns.barplot(data=tmp, x='Category', y='Count', hue='Data')
#ax.set_yticks(ticks=range(0, max(tmp['Count'])+1, 2), labels=range(0, max(tmp['Count'])+1, 2))
#plt.tight_layout()
#plt.savefig('plots/top40_cat_dis.pdf', dpi=300, bbox_inches='tight')
#plt.show()
#plt.close()

# distribution of arXiv categories comparison top 20
from collections import Counter
r = df[df.ranking <= 20]
c3 = Counter(r['primary_category']) #.most_common()
c3.update({'cs.CR': 0, 'cs.AI': 0, 'cs.RO': 0, 'cs.GR': 0})
print(c3)
#raise ValueError

#tmp = collections.defaultdict(list)

c1 = {
    "cs.CL": 12, 'cs.CV': 3, 'cs.LG': 3, 'cs.AI': 1, 'cs.CR': 1, 'cs.GR': 0, 'cs.RO': 0
}


for cat, count in c1.items():
    tmp['Data'].append('arxiv-06/23')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 20')

c2 = {
    "cs.CL": 11, 'cs.CV': 5, 'cs.LG': 3, 'cs.AI': 1, 'cs.CR': 0, 'cs.GR': 0, 'cs.RO': 0
}

for cat, count in c2.items():
    tmp['Data'].append('arxiv-09/23')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 20')

for cat, count in c3.items():
    tmp['Data'].append('arxiv-01/24')
    tmp['Category'].append(cat)
    tmp['Count'].append(count)
    tmp['Top N'].append('Top 20')

tmp = pd.DataFrame(tmp)
#fig, ax = plt.subplots(figsize=(3.5,3))
plt.figure(figsize=(4, 3))
ax = sns.barplot(data=tmp[tmp['Top N']=='Top 20'], x='Category', y='Count', hue='Data', legend=False)
#sns.catplot(data=tmp, kind='bar', x='Category', y='Count', hue='Data', col='Top N', ax=ax)
ax.set_yticks(ticks=range(0, max(tmp['Count'])+1, 2), labels=range(0, max(tmp['Count'])+1, 2))
plt.tight_layout()
plt.savefig('plots/top20_cat_dis.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(4, 3))
ax = sns.barplot(data=tmp[tmp['Top N']=='Top 40'], x='Category', y='Count', hue='Data')
#sns.catplot(data=tmp, kind='bar', x='Category', y='Count', hue='Data', col='Top N', ax=ax)
ax.set_yticks(ticks=range(0, max(tmp['Count'])+1, 2), labels=range(0, max(tmp['Count'])+1, 2))
#ax.set_legned(False)
plt.tight_layout()
plt.savefig('plots/top40_cat_dis.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# which are new in the list?
tmp1 = df[(df.ranking623.isin(['100', 'False']))]
tmp1.to_csv('results/new_623.csv', index=False)
tmp2 = df[(df.ranking923.isin(['100', 'False']))]
tmp2.to_csv('results/new_923.csv', index=False)

print(tmp2[~tmp2.title.isin(tmp1['title'])])
