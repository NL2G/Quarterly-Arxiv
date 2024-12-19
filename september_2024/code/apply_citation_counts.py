import numpy as np
import pandas as pd
import tqdm

def zscore(group, week_start='W-SAT'):
    if group['scholar.citationCount'].max() > 3000:
        print(group['scholar.citationCount'].max())
    mean = group['scholar.citationCount'].mean()  # .values
    std = group['scholar.citationCount'].std(ddof=0)  # .values
    group[week_start] = list((np.array(group['scholar.citationCount']) - mean) / std)
    return group
def compute_zscore(dfc, week_start='W-SAT'):
    dfc.reset_index(drop=True, inplace=True)
    dfc['WeekNumber'] = dfc['published'].dt.to_period(week_start)
    dfc = dfc.groupby('WeekNumber').apply(lambda x: zscore(x, week_start))
    return dfc

if __name__ == '__main__':
    weekdays = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
    files = ['extension2.json']
    df = pd.concat([pd.read_json(f) for f in files])
    df["entry_id"] = [f if not "v" in f.split(".")[-1] else f[:-len(f.split("v")[-1])] for f in df["entry_id"].tolist()]
    df = df.drop_duplicates(subset=["entry_id"])
    #df = df[df["raw_scholar_data"] != None]

    d2 = pd.json_normalize(df['raw_scholar_data'])
    d2.columns = "scholar." + d2.columns
    df.reset_index(drop=True, inplace=True)
    d2.reset_index(drop=True, inplace=True)
    df = pd.concat([df, d2], axis=1)
    df = df[df["scholar.citationCount"] >= 0]

    df['published'] = [pd.to_datetime(date_string, errors='coerce', unit='ms') for date_string in df['published']]
    for week_start in tqdm.tqdm(weekdays, desc="Weekdays"):
        df = compute_zscore(df, week_start)

    zscores = np.array([df[week_start].tolist() for week_start in weekdays])
    df['stable_zscore'] = list(np.mean(zscores, axis=0)-np.std(zscores, axis=0))
    df.sort_values(by='stable_zscore', ascending=False, inplace=True)

    df['year'] = pd.DatetimeIndex(df['published']).year
    df['week'] = df['published'].apply(lambda x: x.to_period('W-SAT').strftime(None))

    df['published'] = df['published'].apply(lambda x: str(x).split('T')[0])
    df = df.reset_index(drop=True)

    df.to_csv("data/outputs/with_z_score.csv", index=False, sep='\t')





