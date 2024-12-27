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



prompt = """Can you evaluate the writing quality of the following text regarding (1) fluency, (2) clarity, (3) grammaticality, (4) readability and (5) coherence?  Please use a Likert scale of 1-5 for each criterion: 1 means the text is not understandable by humans while 5 denotes it's perfect. If the scores don't reach 5, indicate the parts in the text that lower the score. Be brief with your answers.

Answer in the following way:

Fluency score: [1-5]
Fluency issues: [parts in the text that lower the score]

Don't explain in addition to the above.

Text:\n{TEXT}"""

introans = []
absans = []
'''
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
'''

sampledf = pd.read_csv('data/arxiv/analyzed/sample158.tsv', sep='\t').iloc[:100]
absans = []
introans = []
for _, row in tqdm(sampledf.iterrows(), total=len(sampledf)):
    # abstract
    print("\t", row['title'])
    '''
    print("=" * 10)
    print("Abstract: ", row['abstract'])
    ans = api_call(prompt.format(TEXT=row['abstract']))
    absans.append(ans)
    print(ans)
    '''
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

