import demjson
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import (Binarizer, LabelBinarizer, LabelEncoder,
                                   MaxAbsScaler, MinMaxScaler, Normalizer,
                                   OneHotEncoder, RobustScaler, StandardScaler)


class FeaturePreProcessing:
    def __init__(self):
        self.feature_dict = {}

    def _df_transform(self, dataframe: DataFrame, field, name):
        if name in ['Normalizer']:
            return 2, [dataframe[field].values]
        elif name in ['a']:
            return 1, np.transpose(dataframe[field].values)
        else:
            return 1, np.transpose([dataframe[field].values])

    def _field_converse(self, dataframe: DataFrame, fields=None) -> dict:
        if isinstance(fields, str) or isinstance(fields, list):
            new_fields = {}
            for col in dataframe.columns:
                new_fields[col] = fields
            return self._field_converse(dataframe, new_fields)
        elif isinstance(fields, dict):
            for field_k, field_v in fields.items():
                if isinstance(field_v, str):
                    fields[field_k] = {
                        field_v: {}
                    }
                elif isinstance(field_v, list):
                    fields[field_k] = {}
                    for field in field_v:
                        fields[field_k][field] = {}
            return fields
        else:
            print("error")
            return fields

    def _fit(self, dataframe: DataFrame, field_k, field_v=None, *args, **kwargs):
        if field_v in ('stand', 'StandardScaler', None):
            value = StandardScaler(*args, **kwargs)
        elif field_v in ('minmax', 'MinMaxScaler'):
            value = MinMaxScaler(*args, **kwargs)
        elif field_v in ('maxabs', 'MaxAbsScaler'):
            value = MaxAbsScaler(*args, **kwargs)
        elif field_v in ('norm', 'Normalizer'):
            value = Normalizer(*args, **kwargs)
        elif field_v in ('encoder', 'LabelEncoder'):
            value = LabelEncoder()
        elif field_v in ('robust', 'RobustScaler'):
            value = RobustScaler(*args, **kwargs)
        elif field_v in ('Binarizer', 'Binarizer'):
            value = Binarizer(*args, **kwargs)
        elif field_v in ('LabelBinarizer', 'LabelBinarizer'):
            value = LabelBinarizer(*args, **kwargs)
        elif field_v in ('OneHotEncoder', 'OneHotEncoder'):
            value = OneHotEncoder(*args, **kwargs)
        else:
            value = StandardScaler(*args, **kwargs)

        _, df = self._df_transform(dataframe, field_k, field_v)
        value.fit(df, *args, **kwargs)
        if field_k in self.feature_dict.keys():
            self.feature_dict[field_k][field_v] = value
        else:
            self.feature_dict[field_k] = {}
            self.feature_dict[field_k][field_v] = value

    def _transform(self, dataframe: DataFrame, field_k, field_v=None, *args, **kwargs):
        num, df = self._df_transform(dataframe, field_k, field_v)
        if num == 2:
            dataframe[field_k] = self.feature_dict[field_k][field_v].transform(df)[
                0]
        else:
            dataframe[field_k] = self.feature_dict[field_k][field_v].transform(
                df)

    def fit(self, dataframe: DataFrame, fields=None, *args, **kwargs):
        fields = self._field_converse(dataframe, fields)
        for field, params in fields.items():
            for process, v in params.items():
                self._fit(dataframe, field, process, **v)

    def transform(self, dataframe: DataFrame, fields=None, *args, **kwargs):
        fields = self._field_converse(dataframe, fields)
        for field, params in fields.items():
            for process, v in params.items():
                self._transform(dataframe, field, process, **v)
        return dataframe

    def fit_transform(self, dataframe: DataFrame, fields=None, *args, **kwargs):
        fields = self._field_converse(dataframe, fields)
        for field, params in fields.items():
            for process, v in params.items():
                self._fit(dataframe, field, process, **v)
                self._transform(dataframe, field, process, **v)
        return dataframe

    def inverse_transform(self, *args, **kwargs):
        pass


class FeatureDictManage:
    """
    主要作用：将DataFrame中的离散字段映射成递增的数字，
    主要包含两步：
        1. 构建映射关系：构建离散字段->ID的映射
        2. 字段映射：将字段映射到相应的ID
    """

    def __init__(self):
        self.feature_size = {}
        self.feature_map = {}

    @staticmethod
    def _field_converse(dataframe: DataFrame, fields=None) -> dict:
        if fields is None:
            fields = dict(zip(dataframe.columns, dataframe.columns))
        elif isinstance(fields, str):
            fields = {fields: fields}
        elif isinstance(fields, list):
            fields = dict(zip(fields, fields))
        return fields

    def fit(self, dataframe: DataFrame, fields=None):
        """
        构建映射关系
        :param dataframe: 数据
        :param fields: 需要构建映射关系的字段，可以是str,list,map  {data_field : save_field}
        """
        fields = self._field_converse(dataframe, fields)

        for field_k, field_v in fields.items():
            if field_v not in self.feature_size.keys():
                self.feature_map[field_v] = {'': 0}
                self.feature_size[field_v] = 1

            field_list = set(np.array(dataframe[field_k].values.tolist()).reshape(1, -1)[0]) - set(
                self.feature_map.get(field_v, []))
            field_list = list(field_list)
            field_list.sort()
            size = self.feature_size.get(field_v, 0)
            d = dict(
                zip(field_list, [i + size for i in range(len(field_list))]))

            self.feature_map[field_v].update(d)
            self.feature_size[field_v] = len(self.feature_map[field_v])

    def transform(self, dataframe: DataFrame, fields: dict = None) -> DataFrame:
        """
        字段映射
        :param dataframe:
        :param fields:
        :return:
        """
        fields = self._field_converse(dataframe, fields)

        for field_k, field_v in fields.items():
            if field_v not in self.feature_map.keys():
                continue

            if isinstance(dataframe[field_k].values[0], list):
                dataframe[field_k] = dataframe[field_k].apply(
                    lambda x: [self.feature_map[field_v].get(i, 0) for i in x])
            else:
                dataframe[field_k] = dataframe[field_k].apply(
                    lambda x: self.feature_map[field_v][x])
        return dataframe

    def fit_transform(self, dataframe: DataFrame, fields: dict = None) -> DataFrame:
        self.fit(dataframe, fields)
        return self.transform(dataframe, fields)
