from notekeras.backend import keras
from notekeras.component.components import DeepFM
from notekeras.layer import SelfMean, SelfSum, MaskFlatten

Model = keras.models.Model
plot_model = keras.utils.plot_model

# numeric fields
in_score = keras.layers.Input(shape=[1], name="score")  # None*1
in_sales = keras.layers.Input(shape=[1], name="sales")  # None*1

# single value categorical fields
in_gender = keras.layers.Input(shape=[1], name="gender")  # None*1
in_age = keras.layers.Input(shape=[1], name="age")  # None*1

# multiple value categorical fields
in_interest = keras.layers.Input(shape=[3], name="interest")  # None*3, 最长长度3
in_topic = keras.layers.Input(shape=[4], name="topic")  # None*4, 最长长度4

'''
First Order Embeddings
'''
numeric = keras.layers.Concatenate()([in_score, in_sales])  # None*2
dense_numeric = keras.layers.Dense(1)(numeric)  # None*1
emb_gender_1d = keras.layers.Reshape([1])(keras.layers.Embedding(3, 1)(in_gender))  # None*1, 性别取值3种
emb_age_1d = keras.layers.Reshape([1])(keras.layers.Embedding(10, 1)(in_age))  # None*1, 年龄取值10种
emb_interest_1d = keras.layers.Embedding(11, 1, mask_zero=True)(in_interest)  # None*3*1
emb_interest_1d = SelfMean(axis=1)(emb_interest_1d)  # None*1
emb_topic_1d = keras.layers.Embedding(22, 1, mask_zero=True)(in_topic)  # None*4*1
emb_topic_1d = SelfMean(axis=1)(emb_topic_1d)  # None*1

'''compute'''
y_first_order = keras.layers.Add()([dense_numeric,
                                    emb_gender_1d,
                                    emb_age_1d,
                                    emb_interest_1d,
                                    emb_topic_1d])  # None*1

latent = 8
'''Second Order Embeddings'''
emb_score_Kd = keras.layers.RepeatVector(1)(keras.layers.Dense(latent)(in_score))  # None * 1 * K
emb_sales_Kd = keras.layers.RepeatVector(1)(keras.layers.Dense(latent)(in_sales))  # None * 1 * K
emb_gender_Kd = keras.layers.Embedding(3, latent)(in_gender)  # None * 1 * K
emb_age_Kd = keras.layers.Embedding(10, latent)(in_age)  # None * 1 * K
emb_interest_Kd = keras.layers.Embedding(11, latent, mask_zero=True)(in_interest)  # None * 3 * K
emb_interest_Kd = keras.layers.RepeatVector(1)(SelfMean(axis=1)(emb_interest_Kd))  # None * 1 * K
emb_topic_Kd = keras.layers.Embedding(22, latent, mask_zero=True)(in_topic)  # None * 4 * K
emb_topic_Kd = keras.layers.RepeatVector(1)(SelfMean(axis=1)(emb_topic_Kd))  # None * 1 * K

emb = keras.layers.Concatenate(axis=1)([emb_score_Kd,
                                        emb_sales_Kd,
                                        emb_gender_Kd,
                                        emb_age_Kd,
                                        emb_interest_Kd,
                                        emb_topic_Kd])  # None * 6 * K

'''compute'''
summed_features_emb = SelfSum(axis=1)(emb)  # None * K
summed_features_emb_square = keras.layers.Multiply()([summed_features_emb, summed_features_emb])  # None * K

squared_features_emb = keras.layers.Multiply()([emb, emb])  # None * 9 * K
squared_sum_features_emb = SelfSum(axis=1)(squared_features_emb)  # Non * K

sub = keras.layers.Subtract()([summed_features_emb_square, squared_sum_features_emb])  # None * K
sub = keras.layers.Lambda(lambda x: x * 0.5)(sub)  # None * K

y_second_order = SelfSum(axis=1)(sub)  # None * 1

'''deep parts'''
y_deep = MaskFlatten()(emb)  # None*(6*K)
y_deep = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(y_deep))
y_deep = keras.layers.Dropout(0.5)(keras.layers.Dense(64, activation='relu')(y_deep))
y_deep = keras.layers.Dropout(0.5)(keras.layers.Dense(32, activation='relu')(y_deep))
y_deep = keras.layers.Dropout(0.5)(keras.layers.Dense(1, activation='relu')(y_deep))

'''deepFM'''
y = keras.layers.Concatenate(axis=1)([y_first_order, y_second_order, y_deep])
y = keras.layers.Dense(1, activation='sigmoid')(y)

y = DeepFM("deepFm")([emb, y_first_order])
model = Model(inputs=[in_score, in_sales, in_gender, in_age, in_interest, in_topic], outputs=[y])

plot_model(model, 'model3.png', show_shapes=True)
