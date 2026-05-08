import re
import pandas as pd
import numpy as np


def extract_year_from_appearance(text):
    if pd.isna(text):
        return None
    match = re.search(r'\((\d{4})\)', str(text))
    if match:
        year = int(match.group(1))
        if 1930 <= year <= 2030:
            return year
    return None


def normalize_name(name):
    if pd.isna(name):
        return None
    import unicodedata
    name = str(name).strip().title()
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def wide_to_long_marvel(df):
    id_col = df.columns[0]
    appearance_cols = df.columns[1:]
    records = []
    for _, row in df.iterrows():
        character = row[id_col]
        for col in appearance_cols:
            val = row[col]
            year = extract_year_from_appearance(val)
            if year is not None:
                records.append({
                    'character': character,
                    'appearance': val,
                    'year': year
                })
    return pd.DataFrame(records)


def load_ssa(ssa_folder, year_start=1939, year_end=2023):
    import os
    dfs = []
    for fname in sorted(os.listdir(ssa_folder)):
        if not (fname.startswith('yob') and fname.endswith('.txt')):
            continue
        year = int(fname[3:7])
        if not (year_start <= year <= year_end):
            continue
        fpath = os.path.join(ssa_folder, fname)
        df = pd.read_csv(fpath, header=None, names=['name', 'sex', 'count'])
        df['year'] = year
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"No se encontraron archivos SSA en {ssa_folder}."
        )
    return pd.concat(dfs, ignore_index=True)


def cross_lag_correlation(series_a, series_b, max_lag=5, method='pearson'):
    from scipy import stats
    results = []
    for lag in range(0, max_lag + 1):
        a = series_a.copy()
        b = series_b.shift(-lag)
        combined = pd.DataFrame({'a': a, 'b': b}).dropna()
        if len(combined) < 5:
            results.append({'lag': lag, 'correlation': np.nan, 'p_value': np.nan})
            continue
        if method == 'pearson':
            corr, pval = stats.pearsonr(combined['a'], combined['b'])
        else:
            corr, pval = stats.spearmanr(combined['a'], combined['b'])
        results.append({'lag': lag, 'correlation': round(corr, 4), 'p_value': round(pval, 4)})
    return pd.DataFrame(results)


def get_marvel_appearances_by_year(marvel_long_df):
    pivot = (marvel_long_df
             .groupby(['year', 'character'])
             .size()
             .reset_index(name='appearances')
             .pivot(index='year', columns='character', values='appearances')
             .fillna(0))
    return pivot


def get_ssa_name_freq_by_year(ssa_df, name, sex=None):
    mask = ssa_df['name'].str.lower() == name.lower()
    if sex is not None:
        mask &= ssa_df['sex'] == sex
    result = ssa_df[mask].groupby('year')['count'].sum()
    return result