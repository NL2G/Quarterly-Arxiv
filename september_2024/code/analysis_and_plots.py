import os
from random import shuffle
from matplotlib import pyplot as plt
from numpy import median
import pandas as pd
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sympy import Q
import tqdm
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from binoculars import Binoculars
import ast

sns.set_theme()

fig_width = 6.5  # Full page width in inches
fig_height = fig_width * 9 / 16

plt.rcParams.update({
    'font.size': 11,  # General font size
    'axes.titlesize': 12,  # Title font size
    'axes.labelsize': 12,  # X and Y labels font size
    'xtick.labelsize': 11,  # X-axis tick labels font size
    'ytick.labelsize': 11,  # Y-axis tick labels font size
    'legend.fontsize': 11,  # Legend font size
})


class SimpleDetector:
    def __init__(self):
        nltk.download('stopwords')

        # 4 words from Lian et al. Mapping the Increasing Use of LLMs in Scientific Papers
        self.target_words_1 = ['realm', 'intricate', 'showcasing', 'pivotal']

        # Words identified by https://github.com/FareedKhan-dev/Detect-AI-text-Easily/tree/main
        self.target_words_2 = [
            'delve', 'harnessing', 'at the heart of', 'in essence', 'facilitating', 'intrinsic',
            'integral', 'core', 'facet', 'nuance', 'culmination', 'manifestation', 'inherent',
            'confluence', 'underlying', 'intricacies', 'epitomize', 'embodiment', 'iteration',
            'synthesize', 'amplify', 'impetus', 'catalyst', 'synergy', 'cohesive', 'paradigm',
            'dynamics', 'implications', 'prerequisite', 'fusion', 'holistic', 'quintessential',
            'cohesion', 'symbiosis', 'integration', 'encompass', 'unveil', 'unravel', 'emanate',
            'illuminate', 'reverberate', 'augment', 'infuse', 'extrapolate', 'embody', 'unify',
            'inflection', 'instigate', 'embark', 'envisage', 'elucidate', 'substantiate',
            'resonate', 'catalyze', 'resilience', 'evoke', 'pinnacle', 'evolve', 'digital bazaar',
            'tapestry', 'leverage', 'centerpiece', 'subtlety', 'immanent', 'exemplify', 'blend',
            'comprehensive', 'archetypal', 'unity', 'harmony', 'conceptualize', 'reinforce',
            'mosaic', 'catering'
        ]

        self.target_words_3 = ['accentuates', 'achieving', 'acknowledging', 'across', 'additionally', 'address',
                               'addresses', 'addressing', 'adept', 'adhered', 'adhering', 'advancement', 'advancements',
                               'advancing', 'adversarial', 'advocating', 'affirm', 'affirming', 'afflicted', 'aiding',
                               'akin', 'align', 'aligning', 'aligns', 'alongside', 'amid', 'amidst', 'analysis',
                               'announced', 'apologizes', 'approach', 'assess', 'assessments', 'attains', 'augmenting',
                               'avenue', 'avenues', 'based', 'between', 'bolster', 'bolstered', 'bolstering', 'both',
                               'broader', 'burgeoning', 'cal', 'culminating', 'curtailed', 'declare', 'declared',
                               'deductively', 'delineates', 'delve', 'delved', 'delves', 'delving', 'demonstrated',
                               'demonstrates', 'demonstrating', 'dependable', 'despite', 'detrimentally', 'diminishes',
                               'diminishing', 'discern', 'discerned', 'discernible', 'discerning', 'displaying',
                               'distinct', 'distinctions', 'distinctive', 'diverse', 'during', 'easing', 'elevate',
                               'elevates', 'elevating', 'elucidate', 'elucidates', 'elucidating', 'emerged', 'emerges',
                               'emphasises', 'emphasising', 'emphasize', 'emphasizes', 'emphasizing', 'employed',
                               'employing', 'employs', 'empowers', 'emulating', 'enabling', 'encapsulates', 'encompass',
                               'encompassed', 'encompasses', 'encompassing', 'endangering', 'endeavors', 'endeavours',
                               'enduring', 'enhance', 'enhancements', 'enhances', 'enhancing', 'ensuring', 'escalating',
                               'essentials', 'exacerbating', 'exceeding', 'excels', 'exceptional', 'exceptionally',
                               'exhibit', 'exhibited', 'exhibiting', 'exhibits', 'expedite', 'expediting',
                               'exploration', 'explores', 'facilitated', 'facilitating', 'featuring', 'fight',
                               'findings', 'focusing', 'formidable', 'fostering', 'fosters', 'foundational', 'furnish',
                               'furthermore', 'garnered', 'gauged', 'grappling', 'groundbreaking', 'groundwork',
                               'hardest', 'harness', 'harnesses', 'harnessing', 'heightened', 'highlight',
                               'highlighting', 'highlights', 'hinges', 'hinting', 'hold', 'holds', 'however',
                               'illuminates', 'illuminating', 'impact', 'impactful', 'impacting', 'impede', 'impeding',
                               'imperative', 'impressive', 'inadequately', 'including', 'incorporates', 'incorporating',
                               'inherent', 'innovative', 'inquiries', 'insights', 'integrates', 'integrating',
                               'interconnectedness', 'interplay', 'into', 'intricacies', 'intricate', 'intricately',
                               'intriguing', 'introduces', 'involves', 'involving', 'juxtaposed', 'leading',
                               'leverages', 'leveraging', 'merges', 'methodologies', 'meticulous', 'meticulously',
                               'midst', 'multifaceted', 'necessitate', 'necessitates', 'necessitating', 'necessity',
                               'notable', 'notably', 'noteworthy', 'nuanced', 'nuances', 'observed', 'offer',
                               'offering', 'offers', 'optimizing', 'orchestrating', 'outcomes', 'overlooking',
                               'overwhelmed', 'particularly', 'paving', 'pinpoint', 'pinpointed', 'pinpointing',
                               'pioneering', 'pioneers', 'pivotal', 'poised', 'pose', 'posed', 'poses', 'posing',
                               'postponed', 'potential', 'precise', 'predominantly', 'presents', 'pressing',
                               'prevalent', 'primarily', 'promise', 'promising', 'pronounced', 'propelling', 'realizes',
                               'realm', 'realms', 'recognizing', 'refine', 'refines', 'refining', 'reframing',
                               'remains', 'remarkable', 'renowned', 'research', 'resulting', 'rethink', 'revealed',
                               'revealing', 'revolutionize', 'revolutionizing', 'revolves', 'scrutinize', 'scrutinized',
                               'scrutinizing', 'seamless', 'seamlessly', 'seeks', 'serves', 'serving', 'shedding',
                               'sheds', 'showcased', 'showcases', 'showcasing', 'signifying', 'solidify', 'spanned',
                               'spanning', 'specifically', 'spurred', 'stands', 'statement', 'stemming',
                               'strategically', 'strategies', 'streamline', 'streamlined', 'streamlines',
                               'streamlining', 'subsequently', 'substantial', 'substantiated', 'substantiates',
                               'surmount', 'surpass', 'surpassed', 'surpasses', 'surpassing', 'swift', 'their',
                               'thereby', 'these', 'this', 'thorough', 'through', 'transformative', 'ultimately',
                               'uncharted', 'uncovering', 'underexplored', 'underscore', 'underscored', 'underscores',
                               'underscoring', 'understanding', 'unraveling', 'unravels', 'unveil', 'unveiled',
                               'unveiling', 'unveils', 'uphold', 'upholding', 'urging', 'using', 'utilized', 'utilizes',
                               'utilizing', 'valuable', 'various', 'varying', 'verifies', 'versatility', 'wandering',
                               'warranting', 'were', 'while', 'within', 'yielding']

        self.target_words_4 = ['delve', 'underscores', 'delving', 'delves', 'underscored', 'underscoring',
                               'intricacies', 'groundbreaking', 'scrutinizing', 'meticulous', 'realm', 'meticulously',
                               'showcases', 'prowess', 'signifies', 'tapestry', 'intricately', 'bolster', 'intricate',
                               'showcasing', 'dissecting', 'mysteries', 'bolstering', 'unraveling', 'underscore',
                               'nuanced', 'revolutionize', 'transcends', 'elegance', 'glean', 'revolves', 'nuances',
                               'amalgamating', 'endeavors', 'underlining', 'embark', 'revolutionizing', 'scrutinized',
                               'deepen', 'renowned', 'illuminating', 'culminating', 'unearthed', 'solidifies', 'posits',
                               'culmination', 'pivotal', 'tangible', 'unveils', 'unravels', 'endeavor', 'commendable',
                               'profound', 'embarked', 'upholding', 'shines', 'fostering', 'adeptly', 'showcased',
                               'shedding', 'inquiries', 'fascinating', 'feat', 'unravel', 'interconnectedness',
                               'heightened', 'uphold', 'heralding', 'swiftly', 'elucidating', 'groundwork', 'gleaned',
                               'bolsters', 'streamlines', 'comprehending', 'elucidation', 'hinting', 'juxtaposing',
                               'advancing', 'illuminates', 'quest', 'hinted', 'avenues', 'falter', 'underpinnings',
                               'streamlining', 'safeguarding', 'ventures', 'offering', 'strives', 'intriguing',
                               'methodical', 'interrelations', 'mathematicians', 'innovative', 'essence', 'crux',
                               'advancement', 'grappling', 'paving']

        self.target_words = list(set(self.target_words_1 + self.target_words_3))
        print("Number of target words:", len(self.target_words))

        self.stop_words = set(stopwords.words('english'))

    def _preprocess_text(self, text):
        # Simple word split function
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace('\n', ' ').replace('\r', '').strip()
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return filtered_words

    def calculate_frequencies(self, text):
        words = self._preprocess_text(text)
        total_word_count = len(words)

        word_count = Counter(words)
        abs_frequencies = {}
        rel_frquencies = {}

        # Count frequencies of target words
        for target in self.target_words:
            abs_frequencies[target] = word_count[target]
            rel_frquencies[target] = word_count[target] / total_word_count

        return {
            "abs_freq_per_target_word": abs_frequencies,
            "rel_freq_per_target_word": rel_frquencies,
            "abs_freq_total": sum(abs_frequencies.values()),
            "rel_freq_total": sum(abs_frequencies.values()) / total_word_count,
            "total_word_count": total_word_count
        }

        # Recursive function to process files in a folder and its subfolders

    def apply_to_ocr_folder(self, folder_path, output_file="outputs/simple_detect_full.tsv"):
        data = []
        for root, dirs, files in tqdm.tqdm(os.walk(folder_path)):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)

                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        text = text[:int(len(text) * 0.4)]
                        frequencies = self.calculate_frequencies(text)
                        frequencies["filename"] = filename

                        if not "top40" in folder_path:
                            frequencies['date'], frequencies['area'], id = filename.split('___')
                        else:
                            id = filename
                        frequencies['id'] = id[:-3]
                        data.append(frequencies)

        df = pd.DataFrame(data)
        df.to_csv(output_file, sep='\t', index=False)

    def scatterplot(self, df_full, df_top, output_dir="plots", mode="rel_freq_total", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_full = df_full.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()

        df_top = df_top.sort_values(by='date')
        df_top = df_top.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()

        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        plt.plot(df_full['date'], df_full[mode], label="Random papers")
        plt.scatter(df_top['date'], df_top[mode], c="red", label="Top 40 papers")
        plt.xlabel('Date')
        plt.ylabel('Relative LLM word occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=4  # Make the legend span multiple columns
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_plot_{mode}.pdf')
        plt.show()
        plt.close()

    def lineplot_v4(self, df_full, df_top, output_dir="plots", mode="abs_freq_per_target_word", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_full[mode] = df_full[mode].apply(ast.literal_eval)

        # filter the mode dictionary to only include the target words
        tgt = [self.target_words_1, self.target_words_2, self.target_words_3, self.target_words_4]
        for i, target in enumerate(tgt):
            df_full[mode + "_" + str(i)] = df_full[mode].apply(lambda x: {k: v for k, v in x.items() if k in target})
            df_full[mode + "_" + str(i)] = df_full[mode + "_" + str(i)].apply(lambda x: sum(x.values()))

            # df_top[mode + "_" + str(i)] = df_top[mode].apply(lambda x: {k: v for k, v in x.items() if k in target})
            # df_top[mode + "_" + str(i)] = df_top[mode].apply(lambda x: sum(x.values()))

        df_full = df_full.groupby(pd.Grouper(key='date', freq=av))[
            [mode + "_" + str(i) for i in range(4)] + ["total_word_count"]].sum()
        df_full = df_full.reset_index()

        print(df_full.head())
        # df_top = df_top.groupby(pd.Grouper(key='date', freq=av))[[mode + "_" + str(i) for i in range(4)] + ["total_word_count"]].sum()
        # df_top = df_top.reset_index()

        for i in range(4):
            df_full[mode + "_" + str(i)] = (df_full[mode + "_" + str(i)] / df_full["total_word_count"]) / len(tgt[i])
            # df_top[mode + "_" + str(i)] = df_top[mode + "_" + str(i)] / df_top["total_word_count"]

        print(len(df_full))

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        for i, l, style, c in zip(range(4), ["L1", "L2", "L3", "L4"], ['-', '--', '-.', ':'],
                                  ["blue", "orange", "brown", "green"]):
            plt.plot(df_full["date"], df_full[mode + "_" + str(i)], label=l, linestyle=style, linewidth=2.0, c=c)

        vertical_date = "22/11/30"
        plt.axvline(pd.to_datetime(vertical_date, format="%y/%m/%d"), color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Relative LLM word occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=2  # Make the legend span multiple columns
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/line_plot_diff_lists.pdf')
        plt.show()
        plt.close()

    def lineplot_v5(self, df_full, df_top, output_dir="plots", mode="rel_freq_per_target_word", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_full[mode] = df_full[mode].apply(ast.literal_eval)
        df_top = df_top.sort_values(by='date')
        df_top[mode] = df_top[mode].apply(ast.literal_eval)

        monthly_dict1 = {}
        # Loop through each month
        for month in range(1, 12):
            d2 = df_full[(df_full['date'].dt.year.isin([2022])) &
                         (df_full['date'].dt.month == month)]  # Filter for the current month
            a2 = d2[mode].tolist()  # Assuming 'mode' column contains dictionaries

            # Create a temporary dictionary for the current month
            a2_dict = {}
            for i in a2:
                for k, v in i.items():
                    if k in a2_dict:
                        a2_dict[k].append(v)
                    else:
                        a2_dict[k] = [v]

            # Compute the median for the current month and store in the monthly_dicts
            for k in a2_dict.keys():
                a2_dict[k] = sum(a2_dict[k]) / len(a2_dict[k])

            # Store the monthly median dictionary
            monthly_dict1[month] = a2_dict

        # Combine the monthly dictionaries to compute the overall median
        final_dict1 = {}

        # Collect all monthly medians for each key
        for month_dict in monthly_dict1.values():
            for k, v in month_dict.items():
                if k in final_dict1:
                    final_dict1[k].append(v)
                else:
                    final_dict1[k] = [v]

        # Compute the overall median for each key
        for k in final_dict1.keys():
            final_dict1[k] = median(final_dict1[k])

        monthly_dict2 = {}
        # Loop through each month
        for month in range(1, 10):
            d2 = df_full[(df_full['date'].dt.year.isin([2024])) &
                         (df_full['date'].dt.month == month)]  # Filter for the current month
            a2 = d2[mode].tolist()  # Assuming 'mode' column contains dictionaries

            # Create a temporary dictionary for the current month
            a2_dict = {}
            for i in a2:
                for k, v in i.items():
                    if k in a2_dict:
                        a2_dict[k].append(v)
                    else:
                        a2_dict[k] = [v]

            # Compute the median for the current month and store in the monthly_dicts
            for k in a2_dict.keys():
                a2_dict[k] = sum(a2_dict[k]) / len(a2_dict[k])

            # Store the monthly median dictionary
            monthly_dict2[month] = a2_dict

        # Combine the monthly dictionaries to compute the overall median
        final_dict2 = {}

        # Collect all monthly medians for each key
        for month_dict in monthly_dict2.values():
            for k, v in month_dict.items():
                if k in final_dict2:
                    final_dict2[k].append(v)
                else:
                    final_dict2[k] = [v]

        # Compute the overall median for each key
        for k in final_dict2.keys():
            final_dict2[k] = median(final_dict2[k])

        a_dict = {}
        for k in final_dict2.keys():
            try:
                a_dict[k] = final_dict2[k] - final_dict1[k]
            except:
                pass

        # find top 5 words
        tgt = list(dict(sorted(a_dict.items(), key=lambda item: item[1], reverse=True)[:5]).keys())
        shuffle(tgt)
        # tgt = ['realm', 'intricate', 'showcasing', 'pivotal', 'delve']

        for i, target in enumerate(tgt):
            df_full[mode + "_" + str(i)] = df_full[mode].apply(lambda x: {k: v for k, v in x.items() if k == target})
            df_full[mode + "_" + str(i)] = df_full[mode + "_" + str(i)].apply(lambda x: sum(x.values()))
            print(df_full[mode + "_" + str(i)])

            df_top[mode + "_" + str(i)] = df_top[mode].apply(lambda x: {k: v for k, v in x.items() if k == target})
            df_top[mode + "_" + str(i)] = df_top[mode + "_" + str(i)].apply(lambda x: sum(x.values()))

        df_full = df_full.groupby(pd.Grouper(key='date', freq=av))[
            [mode + "_" + str(i) for i in range(5)] + ["total_word_count"]].sum()
        df_full = df_full.reset_index()

        print(df_full.head())
        df_top = df_top.groupby(pd.Grouper(key='date', freq=av))[
            [mode + "_" + str(i) for i in range(5)] + ["total_word_count"]].sum()
        df_top = df_top.reset_index()

        for i in range(5):
            df_full[mode + "_" + str(i)] = (df_full[mode + "_" + str(i)] / df_full["total_word_count"])
            df_top[mode + "_" + str(i)] = df_top[mode + "_" + str(i)] / df_top["total_word_count"]

        print(len(df_full))

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        for i, l, c in zip(range(5), tgt, ["blue", "orange", "green", "red", "brown"]):
            plt.plot(df_full["date"], df_full[mode + "_" + str(i)], label=l, c=c)
            plt.scatter(df_top['date'], df_top[mode + "_" + str(i)], c=c, s=1)

        vertical_date = "22/11/30"
        plt.axvline(pd.to_datetime(vertical_date, format="%y/%m/%d"), color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Relative LLM word occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=1  # Make the legend span multiple columns
        )
        plt.savefig(f'{output_dir}/top_words.pdf')
        plt.show()
        plt.close()

    def lineplot(self, df_full, df_top, output_dir="plots", mode="rel_freq_total", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_cl = df_full[df_full["area"] == "cs_CL"]
        df_lg = df_full[df_full["area"] == "cs_LG"]
        df_cv = df_full[df_full["area"] == "cs_CV"]
        df_ai = df_full[df_full["area"] == "cs_AI"]
        df_cl = df_cl.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        df_lg = df_lg.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        df_cv = df_cv.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        df_ai = df_ai.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()

        df_top = df_top.sort_values(by='date')
        print(df_top.columns)
        dft_cl = df_top[df_top["primary_category"] == "cs.CL"]
        dft_lg = df_top[df_top["primary_category"] == "cs.LG"]
        dft_cv = df_top[df_top["primary_category"] == "cs.CV"]
        dft_ai = df_top[df_top["primary_category"] == "cs.AI"]
        dft_cl = dft_cl.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        dft_lg = dft_lg.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        dft_cv = dft_cv.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()
        dft_ai = dft_ai.groupby(pd.Grouper(key='date', freq=av))[
            ["rel_freq_total", "abs_freq_total"]].median().reset_index()

        print(dft_cv)

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        for df, label, style, c in zip([df_cl, df_lg, df_cv, df_ai], ["CL", "LG", "CV", "AI"], ['-', '--', '-.', ':'],
                                       ["blue", "orange", "brown", "green"]):
            plt.plot(df["date"], df[mode].values, label=label, linewidth=2.0, c=c)

        plt.scatter(dft_cl['date'], dft_cl[mode], c="blue", s=12)
        plt.scatter(dft_lg['date'], dft_lg[mode], c="orange", s=12)
        plt.scatter(dft_cv['date'], dft_cv[mode], c="green", s=12)
        plt.scatter(dft_ai['date'], dft_ai[mode], c="brown", s=12)

        vertical_date = "22/11/30"
        plt.axvline(pd.to_datetime(vertical_date, format="%y/%m/%d"), color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Relative LLM word occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=1  # Make the legend span multiple columns
        )
        plt.savefig(f'{output_dir}/line_plot_{mode}.pdf')
        plt.show()
        plt.close()

    def lineplot_v2(self, df_full, df_top, output_dir="plots", mode="abs_freq_per_target_word", av="M", v="v3"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_full[mode] = df_full[mode].apply(ast.literal_eval)

        if v == "v3":
            # filter the mode dictionary to only include the target words
            df_full[mode] = df_full[mode].apply(
                lambda x: {k: v for k, v in x.items() if k in ['realm', 'intricate', 'showcasing', 'pivotal']})

        df_full[mode] = df_full[mode].apply(lambda x: sum(x.values()))

        df_cl = df_full[df_full["area"] == "cs_CL"]
        df_lg = df_full[df_full["area"] == "cs_LG"]
        df_cv = df_full[df_full["area"] == "cs_CV"]
        df_ai = df_full[df_full["area"] == "cs_AI"]

        df_cl = df_cl.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        df_cl = df_cl.reset_index()
        df_cl[mode] = df_cl[mode] / df_cl["total_word_count"]

        df_lg = df_lg.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        df_lg = df_lg.reset_index()
        df_lg[mode] = df_lg[mode] / df_lg["total_word_count"]

        df_cv = df_cv.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        df_cv = df_cv.reset_index()
        df_cv[mode] = df_cv[mode] / df_cv["total_word_count"]

        df_ai = df_ai.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        df_ai = df_ai.reset_index()
        df_ai[mode] = df_ai[mode] / df_ai["total_word_count"]

        df_top = df_top.sort_values(by='date')
        df_top[mode] = df_top[mode].apply(ast.literal_eval)

        if v == "v3":
            # filter the mode dictionary to only include the target words
            df_top[mode] = df_top[mode].apply(
                lambda x: {k: v for k, v in x.items() if k in ['realm', 'intricate', 'showcasing', 'pivotal']})

        df_top[mode] = df_top[mode].apply(lambda x: sum(x.values()))

        dft_cl = df_top[df_top["primary_category"] == "cs.CL"]
        dft_lg = df_top[df_top["primary_category"] == "cs.LG"]
        dft_cv = df_top[df_top["primary_category"] == "cs.CV"]
        dft_ai = df_top[df_top["primary_category"] == "cs.AI"]

        dft_cl = dft_cl.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        dft_cl = dft_cl.reset_index()
        dft_cl[mode] = dft_cl[mode] / dft_cl["total_word_count"]

        dft_lg = dft_lg.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        dft_lg = dft_lg.reset_index()
        dft_lg[mode] = dft_lg[mode] / dft_lg["total_word_count"]

        dft_cv = dft_cv.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        dft_cv = dft_cv.reset_index()
        dft_cv[mode] = dft_cv[mode] / dft_cv["total_word_count"]

        dft_ai = dft_ai.groupby(pd.Grouper(key='date', freq=av))[[mode, "total_word_count"]].sum()
        dft_ai = dft_ai.reset_index()
        dft_ai[mode] = dft_ai[mode] / dft_ai["total_word_count"]

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        for df, label, style, c in zip([df_cl, df_lg, df_cv, df_ai], ["CL", "LG", "CV", "cs.AI"],
                                       ['-', '--', '-.', ':'], ["blue", "orange", "brown", "green"]):
            plt.plot(df["date"], df[mode].values, label=label, linewidth=2.0, c=c)

        plt.scatter(dft_cl['date'], dft_cl[mode], c="blue", s=12)
        plt.scatter(dft_lg['date'], dft_lg[mode], c="orange", s=12)
        plt.scatter(dft_cv['date'], dft_cv[mode], c="green", s=12)
        plt.scatter(dft_ai['date'], dft_ai[mode], c="brown", s=12)

        vertical_date = "22/11/30"
        plt.axvline(pd.to_datetime(vertical_date, format="%y/%m/%d"), color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Relative LLM word occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=1  # Make the legend span multiple columns
        )
        plt.savefig(f'{output_dir}/line_plot_{v}_{mode}.pdf')
        plt.show()
        plt.close()


class BinocularDetector():
    def __init__(self):
        self.bino = Binoculars()

    # Recursive function to process files in a folder and its subfolders
    def apply_to_ocr_folder(self, folder_path, output_file="outputs/bino_detect_full.tsv"):
        filenames, dates, areas, ids, bino_scores = [], [], [], [], []
        for root, dirs, files in tqdm.tqdm(os.walk(folder_path)):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)

                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        text = text[:int(len(text) * 0.4)]

                        if not "top40" in folder_path:
                            date, area, id = filename.split('___')
                            dates.append(date)
                            areas.append(area)
                        else:
                            id = filename
                            dates.append(None)
                            areas.append(None)

                        ids.append(id[:-3])
                        filenames.append(filename)

                        bino_scores.append(self.bino.compute_score(text))

        with open("bino_scores.txt", "w") as file:
            for score in bino_scores:
                file.write(f"{score}\n")

        df = pd.DataFrame(list(zip(ids, dates, areas, filenames, bino_scores)),
                          columns=["id", "date", "area", "filename", "bino_score"])
        df.to_csv(output_file, sep='\t', index=False)

        return df

    def scatterplot(self, df_full, df_top, output_dir="plots", mode="bino_score", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_full = df_full.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()

        df_top = df_top.sort_values(by='date')
        df_top = df_top.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()

        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        plt.plot(df_full['date'], df_full[mode], label="Random papers")
        plt.scatter(df_top['date'], df_top[mode], c="red", label="Top 40 papers")
        plt.xlabel('Date')
        plt.ylabel('Binoculars Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=1  # Make the legend span multiple columns
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_plot_{mode}.pdf')
        plt.show()
        plt.close()

    def lineplot(self, df_full, df_top, output_dir="plots", mode="bino_score", av="M"):
        # Supported modes: abs_freq_per_target_word, rel_freq_per_target_word, abs_freq_total, rel_freq_total
        df_full = df_full.sort_values(by='date')
        df_cl = df_full[df_full["area"] == "cs_CL"]
        df_lg = df_full[df_full["area"] == "cs_LG"]
        df_cv = df_full[df_full["area"] == "cs_CV"]
        df_ai = df_full[df_full["area"] == "cs_AI"]
        df_cl = df_cl.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        df_lg = df_lg.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        df_cv = df_cv.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        df_ai = df_ai.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()

        df_top = df_top.sort_values(by='date')
        print(df_top.columns)
        dft_cl = df_top[df_top["primary_category"] == "cs.CL"]
        dft_lg = df_top[df_top["primary_category"] == "cs.LG"]
        dft_cv = df_top[df_top["primary_category"] == "cs.CV"]
        dft_ai = df_top[df_top["primary_category"] == "cs.AI"]
        dft_cl = dft_cl.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        dft_lg = dft_lg.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        dft_cv = dft_cv.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()
        dft_ai = dft_ai.groupby(pd.Grouper(key='date', freq=av))["bino_score"].median().reset_index()

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        for df, label, c in zip([df_cl, df_lg, df_cv, df_ai], ["CL", "LG", "CV", "AI"],
                                ["blue", "orange", "brown", "green"]):
            y_smooth = gaussian_filter1d(df[mode].values, sigma=4)  # Adjust `sigma` for more/less smoothing
            plt.plot(df["date"], y_smooth, label=label, c=c)

        plt.scatter(dft_cl['date'], dft_cl[mode], c="blue", s=12)
        plt.scatter(dft_lg['date'], dft_lg[mode], c="orange", s=12)
        plt.scatter(dft_cv['date'], dft_cv[mode], c="brown", s=12)
        plt.scatter(dft_ai['date'], dft_ai[mode], c="green", s=12)

        vertical_date = "22/11/30"
        plt.axvline(pd.to_datetime(vertical_date, format="%y/%m/%d"), color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Binoculars Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(
            loc="upper left",  # Place legend at the top-center below the plot
            ncol=1  # Make the legend span multiple columns
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/line_plot_{mode}.pdf')
        plt.show()
        plt.close()

    def bar(self, df_full, df_top, output_dir="plots", mode="bino_score"):
        df_full = df_full.sort_values(by='date')
        df_top = df_top.sort_values(by='date')

        self.bino.change_mode("accuracy")
        df_full["human"] = self.bino.predict(df_full["bino_score"])
        df_top["human"] = self.bino.predict(df_top["bino_score"])

        print(df_top["human"].value_counts())

        split_date = pd.to_datetime("22/11/30", format="%y/%m/%d")

        df_full['date'] = pd.to_datetime(df_full['date'])
        df_full['period'] = df_full['date'].apply(lambda x: "before ChatGPT" if x <= split_date else "after ChatGPT")

        # Print full where human is False and perid is after ChatGPT
        print(df_full[(df_full["human"] == False) & (df_full["period"] == "after GPT")][["id", "bino_score"]])

        # pd.options.display.float_format = '{:.10f}'.format  # 10 decimal places
        grouped_full = df_full.groupby('period')['human'].value_counts(normalize=False).unstack().fillna(0)
        grouped_full.columns = ['AI Generated', 'Human Written']

        grouped_full = grouped_full.sort_index(ascending=False)

        # grouped_full.plot(kind='bar', figsize=(4, 2.2))

        # Adding percentages to the tick labels
        fig, ax = plt.subplots(figsize=(4, 2.2))

        bar_plot = grouped_full.plot(kind='bar', ax=ax, legend=False)

        ax.set_yscale('log')

        # Adding percentage labels above each bar
        for container in bar_plot.containers:
            ax.bar_label(container, label_type='edge')

        plt.ylabel('Paper Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bar_plot_{mode}.pdf')
        plt.show()


if __name__ == "__main__":
    simple = True
    bino = True

    if simple:
        simple_detector = SimpleDetector()

        # Run detector and save dataframes
        # simple_detector.apply_to_ocr_folder("ocr2")
        # simple_detector.apply_to_ocr_folder("ocr_top40", output_file="outputs/simple_detect_top40.tsv")

        # Load dataframes
        df_full = pd.read_csv("outputs/simple_detect_full.tsv", sep="\t", dtype={"id": str})
        print(len(df_full))
        df_top = pd.read_csv("outputs/simple_detect_top40.tsv", sep="\t", dtype={"id": str})

        # Get dates for top40 ...
        df_src = pd.read_excel('outputs/with_z_score_100_cleaned.xlsx')
        df_src["id"] = df_src["entry_id"].apply(lambda x: x.split("/")[-1]).astype(str)

        df_top = pd.merge(df_top, df_src, how="left", on="id")
        df_top["published"] = pd.to_datetime(df_top["published"])
        df_top["date"] = df_top["published"].dt.strftime('%Y%m%d')

        df_full["date"] = pd.to_datetime(df_full["date"], format='%Y%m%d')
        df_top["date"] = pd.to_datetime(df_top["date"], format='%Y%m%d')

        top_ids = df_top["id"].to_list()

        df_full = df_full[~df_full['id'].isin(top_ids)]

        # Print top 3 "generated" papers from top 40 and from the full list
        print("Largest random paper gen",
              df_full.nlargest(3, 'rel_freq_total')[["id", "rel_freq_total", "abs_freq_total"]])
        print("Largest top40 paper gen",
              df_top.nlargest(3, 'rel_freq_total')[["id", "rel_freq_total", "abs_freq_total"]])

        # Produce Scatteplot
        simple_detector.scatterplot(df_full=df_full, df_top=df_top)
        simple_detector.lineplot(df_full=df_full, df_top=df_top)
        simple_detector.lineplot_v2(df_full=df_full, df_top=df_top)
        simple_detector.lineplot_v2(df_full=df_full, df_top=df_top, v=None)
        simple_detector.lineplot_v4(df_full=df_full, df_top=df_top)
        simple_detector.lineplot_v5(df_full=df_full, df_top=df_top)

    if bino:
        bino_detector = BinocularDetector()
        # bino_detector.apply_to_ocr_folder("ocr2")
        # bino_detector.apply_to_ocr_folder("ocr_top40", output_file="outputs/bino_detect_top40.tsv")

        df_full = pd.read_csv("outputs/bino_detect_full.tsv", sep="\t", dtype={"id": str})
        df_top = pd.read_csv("outputs/bino_detect_top40.tsv", sep="\t", dtype={"id": str})

        print(df_full["date"].value_counts().to_string())

        # Get dates for top40 ...
        df_src = pd.read_excel('outputs/with_z_score_100_cleaned.xlsx')
        df_src["id"] = df_src["entry_id"].apply(lambda x: x.split("/")[-1]).astype(str)

        df_top = pd.merge(df_top, df_src, how="left", on="id")
        df_top["published"] = pd.to_datetime(df_top["published"])
        df_top["date"] = df_top["published"].dt.strftime('%Y%m%d')

        df_full["date"] = pd.to_datetime(df_full["date"], format='%Y%m%d')
        df_top["date"] = pd.to_datetime(df_top["date"], format='%Y%m%d')

        top_ids = df_top["id"].to_list()

        df_full = df_full[~df_full['id'].isin(top_ids)]

        # Print top 3 "generated" papers from top 40 and from the full list

        bino_detector.bar(df_full=df_full, df_top=df_top)

        print("Largest random paper gen", df_full.nlargest(3, 'bino_score')[["id", "bino_score"]])
        print("Largest top40 paper gen", df_top.nlargest(3, 'bino_score')[["id", "bino_score"]])

        df_full["bino_score"] = df_full["bino_score"] * -1
        df_top["bino_score"] = df_top["bino_score"] * -1

        bino_detector.scatterplot(df_full=df_full, df_top=df_top)
        bino_detector.lineplot(df_full=df_full, df_top=df_top)