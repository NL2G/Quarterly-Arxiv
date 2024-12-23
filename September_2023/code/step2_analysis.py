import pandas as pd
df = pd.read_csv("../data/res_citation.csv", sep = "\t")
print(len(df))

def datatime_(x):
    try:
        dt = pd.to_datetime(x).tz_localize(None).to_period('W-SAT')
    except:
        print(x)
    return dt

def preprocess(df):
    # Currently we set weeks to begin sundays. We group by week and calculate the average per week as a new dataframe column
    df["WeekNumber"] = df.published.apply(lambda x: datatime_(x))

    grouped = df.groupby('WeekNumber')
    groups = []
    number = 0
    for name, group in grouped:
        mean = group['citationCount'].mean()
        std = group['citationCount'].std(ddof=0)
        group["z-score"] = (group['citationCount'] - mean) / std
        groups.append(group)
        number = number + len(group)
    print(number)

    # Save dataframe with the normalized citation counts as csv file
    df = pd.concat(groups).sort_values('z-score', ascending=False)
    #df['published'] = df['published'].dt.tz_localize(None)
    #df['updated'] = df['updated'].dt.tz_localize(None)
    df.to_csv("../data/computation_Sept.csv", sep="\t")

preprocess(df)