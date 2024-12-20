import nltk
import pandas as pd
from keybert import KeyBERT
import tqdm
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer


# Read in excel document with the selected top papers
df = pd.read_excel("top_40.xlsx")

# Create list with title, abstract pairs and initialize KeyBERT
titles = df["title"].tolist()
abstracts = df["summary"].tolist()
data = zip(titles, abstracts)
kw_model = KeyBERT()

# Get the top 5 keywords from a concatenation of title+abstract sampled from 200 candidates. Use the specified range of n grams
min_n_gram = 3
max_n_gram = 4
keywords = [
    kw_model.extract_keywords(d[0]+d[1], top_n=5, keyphrase_ngram_range=(n, n), use_mmr=True, diversity=0.6,
                              nr_candidates=200) for
    d in tqdm.tqdm(data) for n in range(min_n_gram, max_n_gram)]

# Use a lemmatizer and filter out specific words, to make the wordclouds more concise. Also use a POS tagger to filter for nouns
lemmatizer = WordNetLemmatizer()
tg = [' '.join([lemmatizer.lemmatize(f) for f in k[0].split(" ")]) for kw in keywords for k in kw if
      not k[0] in ["especially", "specifically", "efficient", "careful", "model", "propose novel", "fine grained",
                   "source", "github com", "like", "recently", "better", "outperforms", "text", "github",
                   "introduce", "parameter", "text", "baseline", "sequence", "entity", "task", "leverage",
                   "corpus generate", "new state"]]
tagged = nltk.pos_tag(tg)
frequencies = nltk.FreqDist([t[0] for t in tagged if (" " in t[0] or t[1] in ["NN", "NNP"]) and not "github" in t[0] and not "pre train" in t[0]])

# Generate a wordcloud from the produced frequency distribution
wc = WordCloud(background_color="white", max_words=2000, width=4000, height=2500)
wc.generate_from_frequencies(frequencies)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.gcf().set_size_inches(40, 40)
plt.savefig('arxiv_top_40_cloud_tri_abstract_title.png', bbox_inches='tight', dpi=500)
