import demjson
from tensorflow import feature_column


def feature_json_parse():
    feature_json = open('test.json', 'r').read()
    feature_json = demjson.decode(feature_json)

    feature_columns = []
    for feature_line in feature_json['tensorTransform']:
        feature_type_name = feature_line['name']
        feature_para = feature_line['parameters']

        if feature_type_name == 'NumericColumn':
            feature_columns.append(feature_column.numeric_column(feature_para['input_tensor']))
        elif feature_type_name == 'BucketizedColumn':
            feature = feature_column.numeric_column(feature_para['input_tensor'])
            feature_columns.append(feature_column.bucketized_column(feature, boundaries=feature_para['boundaries']))
        else:
            print(feature_type_name)


feature_json_parse()
