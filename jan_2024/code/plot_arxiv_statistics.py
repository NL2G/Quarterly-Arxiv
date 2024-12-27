import collections
from collections import  Counter, defaultdict
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/arxiv/arxiv_20240221_with_std_sorted.csv', sep='\t')
df.sort_values('week', ascending=True, inplace=True)
df['Week'] = df['week'].apply(lambda x: x.split('/')[0])

meval = pd.read_csv("data/arxiv/top40-2024_02_21 - 100.tsv", sep='\t')
meval = meval[meval['ranking (or earliest publication time)']=='False']['title']
#print(len(df))
df = df[~df.title.isin(meval)]
#print(len(df))
#raise ValueError

df['category'] = df['primary_category'].apply(lambda x: x.split('.')[0])
#df['subcategory'] = df['primary_category'].apply(lambda x: x.split('.')[0])
print(df)
#labels = sorted(set(df['primary_category']))
#values = [len(df[df.primary_category==l]) for l in labels]
#c = collections.Counter(df['primary_category'])
c = collections.Counter(df['category']).most_common()
print(c)
r = collections.defaultdict(int)
for i, (k, v) in enumerate(c):
    if i >= 5:
        r['others'] += v
    elif k in ['physics', 'math']:
        r['physics+math'] += v
    else:
        r[k] = v

#print(c)
#raise ValueError
#plt.pie(r.values(), labels=r.keys(), autopct='%1.1f%%', pctdistance=.8, explode=[0.0, 0.2, 0.0, 0.1, 0.0], textprops={'fontsize':16}, startangle=-120)
plt.pie(r.values(), labels=r.keys(), autopct='%1.1f%%', pctdistance=.7, explode=[0.0, 0.1, 0.0, 0.25, 0.0], textprops={'fontsize':16}, labeldistance=1)
plt.tight_layout()
#plt.savefig('plots/pie_cat.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
#raise ValueError
c = collections.Counter(df['primary_category']).most_common(10)
r = defaultdict(list)
for k, v in c:
    r['Primary Category'].append(k)
    r['Percentage (%)'].append(v/len(df) * 100)
r = pd.DataFrame(r)
sns.barplot(data=r, x='Primary Category', y='Percentage (%)')
plt.xticks(rotation=45, fontsize=16)
plt.xlabel("")
plt.ylabel("Percentage (%)", fontsize=16)
plt.tight_layout()
#plt.savefig('plots/distribution_top10_cat.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()


'''
for k, v in r.items():
    tmp = df[df.category == k]
    c = collections.Counter(tmp['primary_category'])
    r = collections.defaultdict(int)
    if k == 'cs':
        for i, (k, v) in enumerate(c.items()):
            if i < 10:
                
    print(c)
'''

c = collections.Counter(df['primary_category']).most_common(4)
#c = sorted(c.items(), key=lambda pair: pair[1], reverse=True)
r = defaultdict(list)
print(c)
'''
for i, (k, v) in enumerate(c):
    #print(i)
    #print(k)
    if i < 10:
        tmp = df[df.primary_category == k]
        for week, ttmp in tmp.groupby('week'):
            r['Week'].append(week)
            r['Primary Category'].append(k)
            r['Percentage'].append(len(ttmp)/len(tmp) * 100)
'''
for week, group in df.groupby('week'):
    for k, v in c:
        tmp = group[group.primary_category==k]
        r['Week'].append(week)
        r['Primary Category'].append(k)
        r['Percentage (%)'].append(len(tmp) / len(group) * 100)
        r['Count'].append(len(group))

r = pd.DataFrame(r)


#print(labels)
#raise ValueError
sns.set_style('dark')
plt.figure(figsize=(9,3))

ax = sns.barplot(data=r[r['Primary Category']=='cs.LG'], x='Week', y='Count', legend=False, alpha=0.3)
ax2 = ax.twinx()
sns.lineplot(data=r, x='Week', y='Percentage (%)', hue='Primary Category', ax=ax2)
ax2.legend(loc=3, bbox_to_anchor=(1.07,0.5), borderaxespad=0)

labels = sorted(set(r['Week']))
ax.set_xticks(ticks=range(len(labels)), labels=[l.split('/')[0] for l in labels], rotation=90)
plt.xlim(-1, len(labels))
#print(r)
plt.tight_layout()
#plt.savefig('plots/percent_categories_over_time.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
#raise ValueError
#raise ValueError

sns.set_style('darkgrid')
c = collections.Counter(df['primary_category']).most_common()
#c = sorted(c.items(), key=lambda pair: pair[1], reverse=True)
r = defaultdict(list)
print(c)
labels = ['cs.CV', 'cs.LG', 'cs.CL', 'cs.AI']
c.append(('others', sum([v for k, v in c if k not in labels])))
c = [cc for cc in c if cc[0] in labels+['others']]
print(c)


for week, group in df.groupby('week'):
    last_tmp = 0
    for i, (k, v) in enumerate(c):
        if k != 'others':
            tmp = group[group.primary_category == k]
        else:
            tmp = group[~group.primary_category.isin(labels)]
        r['Week'].append(week)
        r['Primary Category'].append(k)
        r['Count'].append(len(tmp)+last_tmp)
        last_tmp += len(tmp)

plt.figure(figsize=(8,3.5))
r = pd.DataFrame(r)
# check the later periods:
#r = r.iloc[-int(len(r)*0.5):]
for l in ['others', 'cs.AI', 'cs.CL', 'cs.LG', 'cs.CV']:
    sns.barplot(data=r[r['Primary Category']==l], x='Week', y='Count', label=l)
labels = sorted(set(r['Week']))
plt.xticks(ticks=range(len(labels)), labels=[l.split('/')[0] for l in labels], rotation=90)
#plt.xlim(0, len(labels))
plt.tight_layout()
#plt.savefig('plots/count_categories_over_time.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
#plt.show()
#plt.legend()
#print(r)
#raise ValueError
r = defaultdict(list)

cs = ['cs.CV', 'cs.LG', 'cs.CL', 'cs.AI']
for week, group in df.groupby('week'):
    week = week.split('/')[0]
    r['Week'].append(week)
    r['Count'].append(np.mean(group['citationCount']))
    r['Fraction'].append(1)
    r['Std.'].append(np.std(group['citationCount']))
    r['Coefficient of Variation (%)'].append(np.std(group['citationCount']) / np.mean(group['citationCount']) * 100)
    r['Category'].append('Total')
    r['Papers'].append(';'.join(group.sort_values('stable_zscore', ascending=False)[:5]['title']))
    r['CC'].append(';'.join(group.sort_values('stable_zscore', ascending=False)[:5]['primary_category']))

    mean_count = np.mean(group['citationCount'])
    for c in cs:
        tmp = group[group.primary_category==c]
        r['Week'].append(week)
        r['Count'].append(np.mean(tmp['citationCount']))
        r['Fraction'].append(np.mean(tmp['citationCount'])/mean_count)
        r['Std.'].append(np.std(tmp['citationCount']))
        r['Coefficient of Variation (%)'].append(np.std(tmp['citationCount']) / np.mean(tmp['citationCount']) * 100)
        r['Category'].append(c)
        r['Papers'].append(None)
        r['CC'].append(None)

    '''
    r['Week'].append(week)
    r['Count'].append(np.mean(group[~group.primary_category.isin(cs)]['citationCount']))
    r['Std.'].append(np.std(group['citationCount']))
    r['Coefficient of Variation'].append(np.std(group['citationCount'])/ np.mean(group['citationCount']) * 100)
    r['Category'].append('others')
    r['Papers'].append(None)
    r['CC'].append(None)
    '''

r = pd.DataFrame(r)
# check the later periods:
#r = r.iloc[-int(len(r)*0.4):]
print(r)
#r.to_csv('data/arxiv/top5_overtime.csv', index=False)
#r.to_json('data/arxiv/top5_overtime.json', index=False)
sns.set_style('dark')
plt.figure(figsize=(9, 3))
#ax = sns.lineplot(data=r[~r.Category.isin(['Total', 'others'])], x='Week', y='Count', hue='Category')
ax = sns.lineplot(data=r[~r.Category.isin(['Total', 'others'])], x='Week', y='Fraction', hue='Category')
ax2 = ax.twinx()
#sns.barplot(data=r[r.Category=='Total'], x='Week', y='Std.', alpha=0.3)
#sns.barplot(data=r[r.Category=='Total'], x='Week', y='Coefficient of Variation (%)', alpha=0.3)
ax.set_xticks(ticks=range(len(labels)), labels=[l.split('/')[0] for l in labels], rotation=90)
plt.xlim(-1, len(labels))
plt.tight_layout()
plt.savefig('plots/citation_frac_overtime.pdf', dpi=300, bbox_inches='tight')
plt.show()

