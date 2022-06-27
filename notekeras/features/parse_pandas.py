from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('max_colwidth', 500)


def agg_set(size=-1, padding=False):
    def inner(x):
        if size <= 0:
            return list(set(x))
        elif padding is False:
            return list(set(x))[-size:]
        else:
            x = list(set(x))
            return list(x[-size:] + ['0'] * (size - len(x)))

    return inner


def agg_list(size=-1, padding=False):
    # def inner(x):
    #     if size <= 0:
    #         return list(x)
    #     elif padding is False:
    #         return list(x)[-size:]
    #     else:
    #         x = list(x)
    #         return list(x[-size:] + ['0'] * (size - len(x)))
    #
    # return inner

    if size <= 0:
        return lambda x: list(x)
    elif padding is False:
        return lambda x: list(x)[-size:]
    else:
        return lambda x: list(list(x)[-size:] + ['0'] * (size - len(list(x))))


print(agg_list(size=10, padding=True)([1, 2, 5]))


def agg_array(size=-1, pct=True):
    def inner(x):
        category_list = np.zeros(size)
        for category in list(x):
            category_list[int(category) - 1] += 1

        return category_list.tolist()

    def inner2(x):
        category_list = np.zeros(size)
        for category in list(x):
            category_list[int(category) - 1] += 1

        category_list = category_list / category_list.sum()
        category_list = np.round(category_list, 4)
        return category_list.tolist()

    if pct:
        return inner2
    else:
        return inner
    # def inner(x):
    #     if size <= 0:
    #         return 0
    #     category_list = np.zeros(size)
    #     for category in list(x):
    #         category_list[int(category) - 1] += 1
    #
    #     if pct:
    #         category_list = category_list / category_list.sum()
    #         category_list = np.round(category_list, 4)
    #
    #     return category_list.tolist()
    #
    # return inner


def agg_unique_mean():
    def inner(x):
        return np.mean(list(Counter(list(x)).values()))

    return inner


def agg_unique_var():
    def inner(x):
        return np.var(list(Counter(list(x)).values()))

    return inner


def agg_fun_def(agg_name: str, params: dict = None):
    agg_fun_dict = {
        'size': 'size',
        'count': 'count',
        'mean': 'mean',
        'unique': 'nunique',
        'nunique': 'nunique',
        'max': 'max',
        'min': 'min',
        'sum': 'sum',
        'std': 'std',
        'median': 'median',
        'skew': 'skew',
        'list': agg_list,
        'set': agg_set,
        'array': agg_array,
        'unique_mean': agg_unique_mean,
        'unique_var': agg_unique_var,
    }
    if agg_name not in agg_fun_dict:
        return 'size'

    agg_func = agg_fun_dict[agg_name]
    if isinstance(agg_func, str):
        return agg_func
    elif params is None:
        return agg_func()
    else:
        return agg_func(**params)


def config_agg(configs: dict):
    agg_info = {}
    for stat_col in configs['stat_cols'].keys():
        stat_values = configs['stat_cols'][stat_col]

        cols = []
        for stat_value in stat_values:
            col_name = stat_value['col_name']
            agg_func = stat_value['agg_func']
            agg_para = stat_value.get('agg_para', None)

            agg_func = agg_fun_def(agg_func, agg_para)

            cols.append((col_name, agg_func))
        agg_info[stat_col] = cols

    return agg_info


def run():
    path_root = "/Users/liangtaoniu/tmp/dataset/tencent2020/train_preliminary"
    path_ad = path_root + '/ad.csv'
    path_user = path_root + '/user.csv'
    path_click = path_root + '/click_log.csv'

    feature_pandas = {
        "feature1": {
            "group_key": ["user_id"],
            "stat_cols": {
                "time": [
                    {"agg_func": "unique", "col_name": "time_unique"},
                    {"agg_func": "count", "col_name": "time_count"},
                    {"agg_func": "list", "col_name": "time_list", "agg_para": {"size": 10, "padding": True}},
                    {"agg_func": "array", "col_name": "time_pct", "agg_para": {"size": 91, "pct": True}}
                ],
                "advertiser_id": [
                    {"agg_func": "unique", "col_name": "advertiser_unique"},
                    {"agg_func": "count", "col_name": "advertiser_count"},
                    {"agg_func": "list", "col_name": "advertiser_list", "agg_para": {"size": 10, "padding": True}},
                ],
                "creative_id": [
                    {"agg_func": "unique", "col_name": "creative_unique"},
                    {"agg_func": "count", "col_name": "creative_count"},
                    {"agg_func": "list", "col_name": "creative_list", "agg_para": {"size": 10, "padding": True}},
                ],

                "ad_id": [
                    {"agg_func": "unique", "col_name": "ad_unique"},
                    {"agg_func": "count", "col_name": "ad_count"},
                    {"agg_func": "list", "col_name": "ad_list", "agg_para": {"size": 10, "padding": True}},
                ],

                "product_id": [
                    {"agg_func": "unique", "col_name": "product_unique"},
                    {"agg_func": "count", "col_name": "product_count"},
                    {"agg_func": "list", "col_name": "product_list", "agg_para": {"size": 10, "padding": True}},
                ],

                "product_category": [
                    {"agg_func": "unique", "col_name": "category_unique"},
                    {"agg_func": "count", "col_name": "category_count"},
                    {"agg_func": "list", "col_name": "category_list", "agg_para": {"size": 10, "padding": True}},
                    {"agg_func": "array", "col_name": "category_pct", "agg_para": {"size": 18, "pct": True}}
                ],

                "industry": [
                    {"agg_func": "unique", "col_name": "industry_unique"},
                    {"agg_func": "count", "col_name": "industry_count"},
                    {"agg_func": "list", "col_name": "industry_list", "agg_para": {"size": 10, "padding": True}},
                    {"agg_func": "array", "col_name": "industry_pct", "agg_para": {"size": 335, "pct": True}}
                ],
                "weekday": [
                    {"agg_func": "unique", "col_name": "weekday_unique"},
                    {"agg_func": "count", "col_name": "weekday_count"},
                    {"agg_func": "list", "col_name": "weekday_list", "agg_para": {"size": 10, "padding": True}},
                    {"agg_func": "array", "col_name": "weekday_pct", "agg_para": {"size": 7, "pct": True}}
                ]
            }
        }
    }

    feature_pandas2 = {
        "feature1": {
            "group_key": ["user_id"],
            "stat_cols": {
                "creative_id": [
                    {"agg_func": "nunique", "col_name": "creative_unique"},
                    {"agg_func": "count", "col_name": "creative_count"},
                    {"agg_func": "list", "col_name": "creative_list", "agg_para": {"size": 10, "padding": True}}
                ],
                "click_times": [
                    {"agg_func": "nunique", "col_name": "click_unique"},
                    {"agg_func": "count", "col_name": "click_count"},
                    {"agg_func": "list", "col_name": "click_list", "agg_para": {"size": 10, "padding": True}}
                ],
                "time": [
                    {"agg_func": "nunique", "col_name": "time_unique"},
                    {"agg_func": "count", "col_name": "time_count"},
                    {"agg_func": "list", "col_name": "time_list", "agg_para": {"size": 10, "padding": True}},
                    {"agg_func": "array", "col_name": "time_pct", "agg_para": {"size": 99, "pct": True}}
                ],
            }
        }
    }

    df_ad = pd.read_csv(path_ad)
    df_user = pd.read_csv(path_user)
    df_click = pd.read_csv(path_click).head(200 * 100)

    df_ad.loc[df_ad['product_id'] == '\\N', 'product_id'] = 0
    df_ad.loc[df_ad['product_category'] == '\\N', 'product_category'] = 0
    df_ad.loc[df_ad['advertiser_id'] == '\\N', 'advertiser_id'] = 0
    df_ad.loc[df_ad['industry'] == '\\N', 'industry'] = 0

    df_click = df_click.sort_values(['user_id', 'time'])

    pd_merge = pd.merge(df_click, df_ad, on='creative_id')
    pd_merge = pd.merge(pd_merge, df_user, on='user_id')

    pd_merge['weekday'] = pd_merge['time'] % 7 + 1
    pd_merge['product_id'] = pd_merge['product_id'].astype(str)
    pd_merge['product_category'] = pd_merge['product_category'].astype(str)
    pd_merge['advertiser_id'] = pd_merge['advertiser_id'].astype(str)
    pd_merge['industry'] = pd_merge['industry'].astype(str)

    agg_info = config_agg(feature_pandas['feature1'])

    dataframe = pd_merge.groupby(['user_id']).agg(agg_info)
    dataframe = pd.merge(dataframe, df_user, on='user_id')

    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        label1 = tf.keras.backend.one_hot(dataframe.pop('gender'), 2)
        label2 = tf.keras.backend.one_hot(dataframe.pop('age'), 10)

        labels = (label1, label2)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 256
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    train_d = df_to_dataset(train, batch_size=batch_size)
    val_d = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_d = df_to_dataset(test, shuffle=False, batch_size=batch_size)
