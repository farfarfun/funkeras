import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column

from notekeras.backend import layers, backend, plot_model, keras

backend.set_floatx('float32')

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

example_batch = next(iter(train_ds))[0]

fields = [('age', 'int32'),
          ('trestbps', 'int32'),
          ('chol', 'int32'),
          ('thalach', 'int32'),
          ('oldpeak', 'int32'),
          ('slope', 'int32'),
          ('ca', 'int32'),
          ('thal', 'string'),
          ]

# 将源数据的变量输入进来
feature_dict = {}
for field in fields:
    field_name = field[0]
    if field[1] == 'int32':
        field_type = tf.dtypes.int32
    else:
        field_type = tf.dtypes.string

    feature_dict[field_name] = tf.keras.Input((1,), dtype=field_type, name=field_name)

# 对源数据字段进行处理
feature_columns = []
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
feature_columns.append(feature_column.indicator_column(thal))
feature_columns.append(feature_column.embedding_column(thal, dimension=8))

# thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
# feature_columns.append(feature_column.embedding_column(thal_hashed, dimension=100))

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = layers.DenseFeatures(feature_columns)


def local_demo():
    print(len(train), 'train examples')  # 193
    print(len(val), 'validation examples')  # 49
    print(len(test), 'test examples')  # 61

    def demo(feature_col):
        feature_l = layers.DenseFeatures(feature_col)
        print(feature_l(example_batch).numpy())

    thal_one_hot = feature_column.indicator_column(thal)
    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)

    demo(age)
    demo(age_buckets)
    demo(thal_one_hot)
    demo(thal_embedding)
    demo(feature_column.indicator_column(thal_hashed))
    demo(feature_column.indicator_column(crossed_feature))


def seq_test():
    model = tf.keras.Sequential([feature_layer,
                                 layers.Dense(128, activation='relu'),
                                 layers.Dense(128, activation='relu'),
                                 layers.Dense(1, activation='sigmoid')
                                 ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    model.fit(train_ds, validation_data=val_ds, epochs=5)


l0 = feature_layer(feature_dict)
l1 = layers.Dense(128, activation='relu')(l0)
l2 = layers.Dense(128, activation='relu')(l1)
l3 = layers.Dense(1, activation='sigmoid')(l2)

model = keras.models.Model(inputs=list(feature_dict.values()), outputs=[l3])
model.compile(optimizer='adam', loss='binary_crossentropy', )
model.summary()
plot_model(model, to_file='feature.png', show_shapes=True)

model.fit(train_ds, validation_data=val_ds, epochs=5)
