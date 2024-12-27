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


prompt = """Hartley defines 13 types of titles:
1 Titles that announce the general subject, for example:
• The age of adolescence.
• Designing instructional and informational text.
• On writing scientific articles in English.
2 Titles that particularise a specific theme following a general heading, for example:
Pre-writing: The relation between thinking and feeling.
• The achievement of black Caribbean girls: Good practice in Lambeth
schools.
• The role of values in educational research: The case for reflexivity.
3 Titles that indicate the controlling question, for example:
• Is academic writing masculine?
• What is evidence-based practice – and do we want it too?
• What price presentation? The effects of typographic variables on
essay grades.
4 Titles that just state the findings, for example:
• Supramaximal inflation improves lung compliance in patients with
amyotrophic lateral sclerosis.
• Asthma in schoolchildren is greater in schools close to concentrated
animal feeding operations.
• Angiopoetin-2 levels are elevated in exudative pleural effusions.
5 Titles that indicate that the answer to a question will be revealed, for
example:
• Abstracts, introductions and discussions: How far do they differ in
style?
• The effects of summaries on the recall of information.
• Current findings from research on structured abstracts.
6 Titles that announce the thesis – i.e. indicate the direction of the
author’s argument, for example:
• The lost art of conversation.
• Plus ça change . . . Gender preferences for academic disciplines.
• Down with ‘op. cit.’.
7 Titles that emphasise the methodology used in the research, for example:
• Using colons in titles: A meta-analytic review.
• Reading and writing book reviews across the disciplines: A survey
of authors.
• Is judging text on screen different from judging text in print? A
naturalistic email study.
8 Titles that suggest guidelines and/or comparisons, for example:
• Seven types of ambiguity.
• Nineteen ways to have a viva.
• Eighty ways of improving instructional text.
9 Titles that bid for attention by using startling or effective openings, for
example:
• ‘Do you ride an elephant’ and ‘never tell them you’re German’: The
experiences of British Asian, black and overseas student teachers in
the UK.
• Something more to tell you: Gay, lesbian and bisexual young people’s
experiences of secondary schooling.
• Making a difference: An exploration of leadership roles in sixth
form colleges.
10 Titles that attract by alliteration, for example:
• A taxonomy of titles.
• Legal ease and ‘legalese’.
• Referees are not always right: The case of the 3-D graph.
11 Titles that attract by using literary or biblical allusions, for example:
• From structured abstracts to structured articles: A modest proposal.
• Low! They came to pass. The motivations of failing students.
• Lifting the veil on the viva: The experiences of postgraduate students.
12 Titles that attract by using puns, for example:
• Now take this PIL (Patient Information Leaflet).
• A thorn in the Flesch: Observations on the unreliability of computerbased readability formulae (Rudolph Flesch devised a method of
computing the readability of text).
• Unjustified experiments in typographical design (Text set with
equal word-spacing and a ragged right-hand edge is said to be set
‘unjustified’: text set with variable word-spacing and a straight righthand edge is set ‘justified’.)
13 Finally, titles that mystify, for example:
• Outside the whale.
• How do you know you’ve alternated?
• Is October Brown Chinese?

Can you tell me what the type of the following title is? Please answer in the following way:

Type: [1-13]
Explain: [briefly explain why you think so]

Title: {TEXT}"""


#absans = []
#introans = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    # abstract
    print("\t", row['title'])
    ans = api_call(prompt.format(TEXT=row['title']))
    print(ans)
    print("="*20)
    #raise ValueError


sampledf = pd.read_csv('data/arxiv/analyzed/sample158.tsv', sep='\t').iloc[:100]
for _, row in tqdm(sampledf.iterrows(), total=len(sampledf)):
    # abstract
    print("\t", row['title'])
    ans = api_call(prompt.format(TEXT=row['title']))
    print(ans)
    print("="*20)


