import numpy as np
import pandas as pd
from notekeras.features import FeatureDictManage
from pandas import DataFrame

if __name__ == '__main__':
    feature_dict = FeatureDictManage()

    col_num = 4
    df = pd.DataFrame(np.random.random([10, col_num]), columns=[
                      'C{}'.format(i) for i in range(col_num)])
    feature_dict.add_feature(df, {'C0': 'C2'})
    feature_dict.add_feature(df, 'C1')

    df['C0'] = df['C0'].apply(lambda x: str(round(x * 3)))

    feature_dict.add_feature(df, {'C0': 'C2'})
    print(df)
    print(feature_dict.feature_map)
    print(feature_dict.feature_size)
