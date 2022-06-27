import logging

import demjson
import numpy as np
import pandas as pd
import pyspark.sql.functions as func
import tensorflow as tf
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.types import StringType
from tensorflow import keras
from tensorflow.keras import layers

from notekeras.component.transformer import EncoderList
from notekeras.feature import ParseFeatureConfig
from notetool.load.core import DataLoadAndSave

pd.set_option('max_colwidth', 500)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("working")
logger.setLevel(logging.INFO)

# 对数据进行切片，专业名词叫shuffle。因为我读取的数据是20g，所以设置成3000份，每次每个进程（线程）读取一个shuffle，避免内存不足的异常。
spark = (SparkSession
         .builder
         .appName("taSpark")
         .config("spark.driver.memory", "15g")
         .config("spark.default.parallelism", 1000)
         .config('spark.jars.packages', 'org.tensorflow:spark-tensorflow-connector_2.11:1.15.0')
         .getOrCreate())


class MyData:
    def __init__(self, path_root):
        self.path_root = path_root
        self.data_load = DataLoadAndSave(path_dir=path_root + 'data')

        self.ad_df = None
        self.user_df = None
        self.click_df = None
        self.merge_df = None
        self.dataframe = None

    def build(self, limit=None):
        path_ad = self.path_root + '/ad.csv'
        path_user = self.path_root + '/user.csv'
        path_click = self.path_root + '/click_log.csv'

        self.ad_df = spark.read.csv(path_ad, header=True, inferSchema="true")
        self.user_df = spark.read.csv(path_user, header=True, inferSchema="true")
        self.click_df = spark.read.csv(path_click, header=True, inferSchema="true")
        if limit is not None:
            self.click_df = self.click_df.limit(limit)

    def build_merge(self):
        for col_name in ['product_id', 'creative_id', 'ad_id', 'product_category', 'advertiser_id', 'industry']:
            self.ad_df = self.ad_df.withColumn(col_name,
                                               func.udf(lambda x: '0' if x == '\\N' else str(x), StringType())(
                                                   col_name))

        merge_df = self.click_df.join(self.ad_df, on='creative_id').join(self.user_df, on='user_id')
        merge_df = merge_df.withColumn("weekday", merge_df.time % 7)

        for key in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'weekday', 'creative_id']:
            merge_df = merge_df.withColumn(key, merge_df[key].cast(StringType()))
        self.merge_df = merge_df

    def get_word_vec(self):
        data = self.merge_df.groupBy('user_id').agg(
            func.sort_array(func.collect_list(func.struct(func.col('time'), func.col('ad_id'))), asc=True).alias(
                'items'))
        data = data.withColumn("items", func.udf(lambda x: [i[1] for i in x], ArrayType(StringType()))('items'))

        word2Vec = Word2Vec(vectorSize=128, minCount=10, inputCol="items", outputCol="result")
        model = word2Vec.fit(data.repartition(1000))
        return model

    def embedding_fields(self):
        def word2vec_embedding(model):
            vocab_list = model.getVectors().rdd.map(lambda x: x[0]).collect()
            embeddings = model.getVectors().rdd.map(lambda x: x[1].array).collect()

            # vocab_list.insert(0,'default')
            embeddings.insert(0, np.zeros(len(embeddings[0])))
            return vocab_list, np.array(embeddings)

        for col, key in [('product_id', 'product'), ('product_category', 'category'), ('advertiser_id', 'advertiser'),
                         ('industry', 'industry'), ('creative_id', 'creative'), ('ad_id', 'ad')]:
            model = self.get_word_vec()
            vocab_list, embeddings = word2vec_embedding(model)
            pd.DataFrame(vocab_list).to_csv(self.path_root + '/data/{}.vocab'.format(key), header=None, index=None)
            self.data_load.save(embeddings, filename='{}.embedding'.format(key))

            logger.info(col + " done")

    def build_dataset(self):
        logger.info('doing')
        path = self.path_root + '/feature-pandas.json'
        configs = demjson.decode(open(path, 'r').read())

        aggs = []
        pass_keys = []  # ['time','advertiser_id','creative_id','weekday','product_id','industry']
        for config in configs['feature1']['stat_cols2']:
            col = config['col']
            col_name = config['col_name']
            agg_func = config['agg_func']
            if col in pass_keys:
                continue
            if agg_func == 'count':
                aggs.append(func.count(col).alias(col_name))
            elif agg_func == 'unique':
                aggs.append(func.countDistinct(col).alias(col_name))
            elif agg_func == 'set':
                aggs.append(func.collect_set(col).alias(col_name))
            elif agg_func == 'list':
                aggs.append(
                    func.sort_array(func.collect_list(func.struct(func.col('time'), func.col(col))), asc=True).alias(
                        col_name))
            else:
                print(config)
        # print(aggs)
        df = self.merge_df.groupBy(configs['feature1']['group_key']).agg(*aggs)
        for config in configs['feature1']['stat_cols2']:

            col = config['col']
            col_name = config['col_name']
            agg_func = config['agg_func']
            if col in pass_keys:
                continue
            if agg_func == 'list':
                df = df.withColumn(col_name, func.udf(lambda x: [i[1] for i in x], ArrayType(StringType()))(col_name))
            if agg_func in ('list', 'set'):
                df = df.withColumn(col_name, func.udf(lambda x: list(list(x)[-20:] + ['0'] * (20 - len(list(x)))),
                                                      ArrayType(StringType()))(col_name))
        df = df.join(self.user_df, on='user_id')
        self.dataframe = df.cache()
        logger.info('done')

    def build_train(self, train=True):
        col_name = [
            'time_unique', 'time_count', 'time_list', 'advertiser_unique', 'advertiser_count', 'advertiser_list',
            'creative_unique', 'creative_count', 'creative_list', 'ad_unique', 'ad_count', 'ad_list',
            'product_unique', 'product_count', 'product_list', 'category_unique', 'category_count', 'category_list',
            'industry_unique', 'industry_count', 'industry_list', 'weekday_unique', 'weekday_count', 'weekday_list',
            # 'age','gender'
        ]
        output_types = {}
        output_shapes = {}
        for col in col_name:
            if col == 'age':
                output_types[col] = tf.float32
                output_shapes[col] = (2,)
            elif col == 'gender':
                output_types[col] = tf.float32
                output_shapes[col] = (10,)
            elif '_list' in col:
                output_types[col] = tf.string
                output_shapes[col] = (20)
            else:
                output_types[col] = tf.float32
                output_shapes[col] = (1,)
        output_shapes = (output_shapes, ((2,), (10,)))
        output_types = (output_types, (tf.float32, tf.float32))

        def data_g(df):
            def data_generator():
                for d in df.toJSON().toLocalIterator():
                    d = eval(d)
                    res = {}
                    for col in col_name:
                        if isinstance(d[col], list):
                            res[col] = np.array(d[col])
                        else:
                            res[col] = [d[col]]

                    age = tf.keras.backend.one_hot(d['age'], 2)
                    gender = tf.keras.backend.one_hot(d['gender'], 10)

                    yield res, (age, gender)

            return data_generator

        print(output_shapes)
        print(output_types)
        if train:
            df_train, df_test = self.dataframe.randomSplit([0.9, 0.1])
            self.train_d = tf.data.Dataset.from_generator(data_g(df_train), output_types=output_types,
                                                          output_shapes=output_shapes).batch(512)
            self.val_d = tf.data.Dataset.from_generator(data_g(df_test), output_types=output_types,
                                                        output_shapes=output_shapes).batch(512)
        else:
            self.train_d = tf.data.Dataset.from_generator(self.dataframe, output_types=output_types,
                                                          output_shapes=output_shapes).batch(512)
        # self.train_d, self.val_d = s.randomSplit(0.9, 0.1)

    def get_feature_json(self):
        feature_json = open(self.path_root + 'test.json', 'r').read()
        feature_json = demjson.decode(feature_json)
        return feature_json

    def train_model(self):
        parse = ParseFeatureConfig()
        feature_json = self.get_feature_json()

        layer1 = parse.parse_feature_json(feature_json['layer1'])

        layer_creative, _ = parse.parse_sequence_feature_json(feature_json['layer-creative'])
        layer_creative = EncoderList(encoder_num=1, head_num=16, hidden_dim=5, name='creative-encode')(layer_creative)
        layer_creative = layers.Conv1D(100, 4, activation='relu')(layer_creative)
        layer_creative = layers.GlobalAveragePooling1D()(layer_creative)
        # layer_creative = tf.keras.backend.mean(layer_creative, axis=1)

        layer_product, _ = parse.parse_sequence_feature_json(feature_json['layer-product'])
        layer_product = EncoderList(encoder_num=1, head_num=16, hidden_dim=5, name='product-encode')(layer_product)
        layer_product = layers.Conv1D(100, 4, activation='relu')(layer_product)
        layer_product = layers.GlobalAveragePooling1D()(layer_product)
        # layer_product = tf.keras.backend.mean(layer_product, axis=1)

        layer_category, _ = parse.parse_sequence_feature_json(feature_json['layer-category'])
        layer_category = EncoderList(encoder_num=1, head_num=16, hidden_dim=5, name='cate-encode')(layer_category)
        layer_category = layers.Conv1D(100, 4, activation='relu')(layer_category)
        layer_category = layers.GlobalAveragePooling1D()(layer_category)
        # layer_category = tf.keras.backend.mean(layer_category, axis=1)

        layer_advertiser, _ = parse.parse_sequence_feature_json(feature_json['layer-advertiser'])
        layer_advertiser = EncoderList(encoder_num=1, head_num=16, hidden_dim=5, name='advertiser-encode')(
            layer_advertiser)
        layer_advertiser = layers.Conv1D(100, 4, activation='relu')(layer_advertiser)
        layer_advertiser = layers.GlobalAveragePooling1D()(layer_advertiser)
        # layer_advertiser = tf.keras.backend.mean(layer_advertiser, axis=1)

        layer_industry, _ = parse.parse_sequence_feature_json(feature_json['layer-industry'])
        layer_industry = EncoderList(encoder_num=1, head_num=16, hidden_dim=5, name='industry-encode')(layer_industry)
        layer_industry = layers.Conv1D(100, 4, activation='relu')(layer_industry)
        layer_industry = layers.GlobalAveragePooling1D()(layer_industry)
        # layer_industry = tf.keras.backend.mean(layer_industry, axis=1)

        # l0=layer1
        l0 = keras.backend.concatenate(
            [layer1, layer_category, layer_advertiser, layer_industry, layer_product, layer_creative])

        l0 = layers.Dropout(0.2)(l0)

        l1 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(l0)

        l3 = layers.Dense(2, activation='softmax', name='gender', kernel_regularizer=keras.regularizers.l2(0.01))(l1)

        l4 = layers.Dense(10, activation='softmax', name='age', kernel_regularizer=keras.regularizers.l2(0.01))(l1)

        optimizer = tf.keras.optimizers.Adam()
        model = keras.models.Model(inputs=list(parse.feature_dict.values()), outputs=[l3, l4])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1, min_lr=0.00001)
        model.fit(self.train_d, validation_data=self.val_d, epochs=100, callbacks=[reduce_lr])

    def run(self):
        self.build()
        self.build_merge()
        # self.embedding_fields()
        self.build_dataset()
        self.build_train()
        self.train_model()
