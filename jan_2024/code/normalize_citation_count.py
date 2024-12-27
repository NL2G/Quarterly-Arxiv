import collections

import pandas as pd
import numpy as np


def compute_zscore(df, week_start='W-SAT'):
    df['WeekNumber'] = df['published'].dt.to_period(week_start)
    grouped = df.groupby('WeekNumber')
    zscores = []
    for name, group in grouped:
        mean = group['citationCount'].mean()#.values
        std = group['citationCount'].std(ddof=0)#.values
        zscores += list((np.array(group['citationCount']) - mean) / std)
    return zscores


if __name__ == '__main__':
    #file = 'data/arxiv/arxiv_20230101-20240131_20240221_71139.csv'
    file = "data/arxiv/arxiv_202402010000_202405150000.csv"
    df = pd.read_csv(file, delimiter='\t')
    df['published'] = [pd.to_datetime(date_string, errors='coerce') for date_string in df['published']]
    zscores = []
    for week_start in ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']:
        zscores.append(compute_zscore(df, week_start))

    zscores_dict = {}
    for i, week_start in enumerate(['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']):
        zscores_dict[week_start] = zscores[i]
    zscores = np.array(zscores)
    print(zscores)
    mean_zscores = np.mean(zscores, axis=0)
    print(mean_zscores.shape)
    std_zscores = np.std(zscores, axis=0)
    print(std_zscores.shape)
    stable_zscores = mean_zscores - std_zscores
    print(stable_zscores.shape)
    zscores_dict['stable'] = list(stable_zscores)


    df['stable_zscore'] = list(stable_zscores)
    df.sort_values(by='stable_zscore', ascending=False, inplace=True)

    df['year'] = pd.DatetimeIndex(df['published']).year
    #df['week'] = df['published'].dt.isocalendar().week
    df['week'] = df['published'].apply(lambda x: x.to_period('W-SAT').strftime(None))
    print(sorted(set(df['week'])))

    df['published'] = df['published'].apply(lambda x: str(x).split('T')[0])
    df = df[['title', 'primary_category', 'entry_id', 'published', 'week', 'citationCount', 'stable_zscore']]

    #df.to_csv('data/arxiv/arxiv_20240221_with_std_sorted.csv', sep='\t', index=False)
    #df.to_csv('data/arxiv/arxiv_20240221_no_std_sorted.csv', sep='\t', index=False)
    #df.head(100).to_csv('data/arxiv/arxiv_20240221_top100_with_std.csv', sep='\t', index=False)
    #df.head(100).to_csv('data/arxiv/arxiv_20240221_top100_no_std.csv', sep='\t', index=False)

    df.to_csv(file.replace('.csv', '_zscore_sorted.csv'), index=False)




