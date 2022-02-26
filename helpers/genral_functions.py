import pandas as pd
from helpers.constants import ISRAELI_DATA_PATH, FILTER, ISRAEL_OFFICIAL_DATA_PATH


def mask_words(string_series, pattern, include=False):
    """
     include=True -> keep all foods that include the re pattren
     include=False -> remove all foods that include the re pattren
    """
    mask = string_series.str.contains(pattern, case=False, na=False)
    return mask if include else ~mask


def prepare_data():
    df = pd.read_csv(ISRAELI_DATA_PATH)
    mask = mask_words(df['shmmitzrach'], FILTER, include=False)
    df = df[mask]
    """
    names = df[['shmmitzrach']].values
    with open("filter.txt", 'w') as f:
        for name in names.tolist():
            f.write(name[0])
            f.write('\n')
    """
    return df
