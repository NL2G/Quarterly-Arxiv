import collections
import json,sys
import openai
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

#prompt = "Who is president of US?"

client = OpenAI(api_key=openai.api_key)


def api_call(prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# top40
df = pd.read_csv('data/arxiv/analyzed/top40.tsv', sep='\t')



prompt = """Does the following text adhere to the SMART principle? If so, please indicate the parts in the text it does so, seperately for each SMART criterion. Please ignore the "goal" aspect of SMART but focus on the writing style. Be brief with your answer.

Answer in the following way: 
SMART: [yes/no]
Evidence_specific: [yes/no; the evidence text in the given text for criterion "specific"]
Evidence_measurable: [yes/no; the evidence text in the given text for criterion "measurable"]
...

Text:{TEXT}"""

absans = []
introans = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    # abstract
    print("\t", row['title'])
    print("=" * 10)
    print("Abstract: ", row['abstract'])
    ans = api_call(prompt.format(TEXT=row['abstract']))
    absans.append(ans)
    print(ans)
    #print(type(row['introduction']))
    if isinstance(row['introduction'], str):
        print("=" * 10)
        print('Introduction: ', row['introduction'])
        ans = api_call(prompt.format(TEXT=row['introduction']))
        print(ans)
        introans.append(ans)
    else:
        introans.append(None)
    print("="*20)
    #raise ValueError

df['abstract_smart'] = absans
df['introduction_smart'] = introans

df.to_csv('data/arxiv/analyzed/top40_smart.tsv', sep='\t', index=False)

sampledf = pd.read_csv('data/arxiv/analyzed/sample158.tsv', sep='\t').iloc[:100]
absans = []
introans = []
for _, row in tqdm(sampledf.iterrows(), total=len(sampledf)):
    # abstract
    print("\t", row['title'])
    print("=" * 10)
    print("Abstract: ", row['abstract'])
    ans = api_call(prompt.format(TEXT=row['abstract']))
    absans.append(ans)
    print(ans)
    #print(type(row['introduction']))
    if isinstance(row['introduction'], str):
        print("=" * 10)
        print('Introduction: ', row['introduction'])
        ans = api_call(prompt.format(TEXT=row['introduction']))
        print(ans)
        introans.append(ans)
    else:
        introans.append(None)
    print("="*20)
    #raise ValueError

sampledf['abstract_smart'] = absans
sampledf['introduction_smart'] = introans

sampledf.to_csv('data/arxiv/analyzed/sample100_smart.tsv', sep='\t', index=False)
