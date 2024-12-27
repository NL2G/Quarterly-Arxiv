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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# top40
df = pd.read_csv('data/arxiv/analyzed/top40.tsv', sep='\t')



prompt = """A good introduction or abstract of a paper should fulfill 3 moves:  (a) establishing a territory (why the research is important); (b) outlining a niche, i.e., weaknesses and gaps in existing research, and (c) occupying that niche, i.e., providing novel solutions. Can you let me know if the following text adherse to the 3 moves? If so, please indicate the parts in the text that do so.

Answer in the following way: 
3 moves: [yes/no]
Evidence a: [evidence text in the abstract/introduction for move (a)]
Evidence b: [evidence text in the abstract/introduction for move (b)]
Evidence c: [evidence text in the abstract/introduction for move (c)]

Text:{TEXT}"""

introans = []
absans = []

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

