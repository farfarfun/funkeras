import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Dense,
                                     Dropout, Embedding, Flatten, Input, Layer,
                                     ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from ...layers import DNN
from ...layers import FFM as FFM_Layer
from ...layers import CrossLayer as CrossNetwork
from ...layers import Dice, Linear
from ...layers.fm import FactorizationMachine


class AFM(Model):
    def __init__(self, feature_columns, mode, activation='relu', embed_reg=1e-4):
        """
        AFM 
        :param feature_columns: A list. dense_feature_columns and sparse_feature_columns
        :param mode:A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param activation: A string. Activation function of attention.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        if self.mode == 'att':
            t = (len(self.embed_layers) - 1) * len(self.embed_layers) // 2
            self.attention_W = Dense(units=t, activation=activation)
            self.attention_dense = Dense(units=1, activation=None)

        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](
            sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # (None, len(sparse_inputs), embed_dim)
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])
        # Pair-wise Interaction Layer
        # for loop is badly
        element_wise_product_list = []
        # t = (len - 1) * len /2, k = embed_dim
        for i in range(embed.shape[1]):
            for j in range(i+1, embed.shape[1]):
                element_wise_product_list.append(
                    tf.multiply(embed[:, i], embed[:, j]))
        element_wise_product = tf.transpose(
            tf.stack(element_wise_product_list), [1, 0, 2])  # (None, t, k)
        # mode
        if self.mode == 'max':
            x = tf.reduce_sum(element_wise_product, axis=1)   # (None, k)
        elif self.mode == 'avg':
            x = tf.reduce_mean(element_wise_product, axis=1)  # (None, k)
        else:
            x = self.attention(element_wise_product)  # (None, k)
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def attention(self, keys):
        a = self.attention_W(keys)  # (None, t, t)
        a = self.attention_dense(a)  # (None, t, 1)
        a_score = tf.nn.softmax(a)  # (None, t, 1)
        a_score = tf.transpose(a_score, [0, 2, 1])  # (None, 1, t)
        outputs = tf.reshape(tf.matmul(a_score, keys),
                             shape=(-1, keys.shape[2]))  # (None, k)
        return outputs


class NFM(Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=1e-4):
        """
        NFM architecture
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param bn_use: A Boolean. Use BatchNormalization or not.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1)

    def call(self, inputs):
        # Inputs layer
        dense_inputs, sparse_inputs = inputs
        # Embedding layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        # (None, filed_num, embed_dim)
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        # Bi-Interaction Layer
        embed = 0.5 * (tf.pow(tf.reduce_sum(embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(embed, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = tf.concat([dense_inputs, embed], axis=-1)
        # BatchNormalization
        x = self.bn(x, training=self.bn_use)
        # Hidden Layers
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs


class CIN(Layer):
    """
    CIN part
    """

    def __init__(self, cin_size, l2_reg=1e-4):
        """

        :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
        :param l2_reg: A scalar. L2 regularization.
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.embedding_nums = input_shape[1]
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(
                    1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        # dim * (None, field_nums[0], 1)
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        for idx, size in enumerate(self.cin_size):
            # dim * (None, filed_nums[i], 1)
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)

            # (dim, None, field_nums[0], field_nums[i])
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)

            result_2 = tf.reshape(
                result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            # (None, dim, field_nums[0] * field_nums[i])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            # (None, field_num[i+1], dim)
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        # (None, H_1 + ... + H_K, dim)
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)

        return result


class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-5, cin_reg=1e-5):
        """
        xDeepFM
        :param feature_columns: A list. a list containing dense and sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. the regularizer of embedding.
        :param cin_reg: A scalar. the regularizer of cin.
        """
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.linear = Linear()
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units,
                       dropout=dnn_dropout, activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(
            1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # linear  delete
        # linear_out = self.linear(sparse_inputs)

        embed = [self.embed_layers['embed_{}'.format(i)](
            sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # cin
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (None, embedding_nums, dim)
        cin_out = self.cin_dense(cin_out)
        # dnn
        embed_vector = tf.reshape(
            embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)

        output = tf.nn.sigmoid(cin_out + dnn_out + self.bias)
        return output

    def get_config(self):
        pass


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, implicit=False, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Layer
        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param implicit: whether implicit or not
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.implicit = implicit
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        keras.layers.Embedding
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='user_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs, **kwargs):
        user_id, item_id, avg_score = inputs
        # MF
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id)
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id)
        outputs = tf.reduce_sum(tf.multiply(
            latent_user, latent_item), axis=1, keepdims=True)
        # MF-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id)
        bias = tf.reshape((avg_score + user_bias + item_bias), shape=(-1, 1))
        # use bias
        outputs = bias + outputs if self.use_bias else outputs
        # implicit expression dataset
        if self.implicit:
            outputs = tf.nn.sigmoid(outputs)
        return outputs


class MF(Model):
    def __init__(self, feature_columns, implicit=False, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param implicit: whether implicit or not
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        num_users, num_items = self.sparse_feature_columns[0]['feat_num'], \
            self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        self.mf_layer = MF_layer(num_users, num_items, latent_dim, implicit, use_bias,
                                 user_reg, item_reg, user_bias_reg, item_bias_reg)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        avg_score = dense_inputs
        outputs = self.mf_layer([user_id, item_id, avg_score])
        return outputs


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        """
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)

        # Wide
        wide_out = self.linear(dense_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * (wide_out + deep_out))
        return outputs


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4, cross_w_reg=1e-4, cross_b_reg=1e-4):
        """
        Deep&Cross Network
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cross_w_reg: A scalar. The regularizer of cross network.
        :param cross_b_reg: A scalar. The regularizer of cross network.
        """
        super(DCN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(
            self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs


class PNN(Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-4, w_z_reg=1e-4, w_p_reg=1e-4, l_b_reg=1e-4):
        """
        Product-based Neural Networks
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param mode: A string. 'in' IPNN or 'out'OPNN.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_z_reg: A scalar. The regularizer of w_z_ in product layer
        :param w_p_reg: A scalar. The regularizer of w_p in product layer
        :param l_b_reg: A scalar. The regularizer of l_b in product layer
        """
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num,
                                          self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num,
                                              self.field_num, hidden_units[0]),
                                       initializer='random_uniform',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        # out
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.embed_dim,
                                              self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = DNN(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        # (None, field_num, embed_dim)
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        z = embed  # (None, field, embed_dim)
        # product layer
        if self.mode == 'in':
            # (None, field_num, field_num)
            p = tf.matmul(embed, tf.transpose(embed, [0, 2, 1]))
        else:  # out
            # (None, 1 embed_num)
            f_sum = tf.reduce_sum(embed, axis=1, keepdims=True)
            # (None, embed_num, embed_num)
            p = tf.matmul(tf.transpose(f_sum, [0, 2, 1]), f_sum)

        l_z = tf.tensordot(z, self.w_z, axes=2)  # (None, h_unit)
        l_p = tf.tensordot(p, self.w_p, axes=2)  # (None, h_unit)
        l_1 = tf.nn.relu(
            tf.concat([l_z + l_p + self.l_b, dense_inputs], axis=-1))
        # dnn layer
        dnn_x = self.dnn_network(l_1)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs


class Residual_Units(Layer):
    """
    Residual Units
    """

    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: A list. Neural network hidden units.
        :param dim_stack: A scalar. The dimension of inputs unit.
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class Deep_Crossing(Model):
    def __init__(self, feature_columns, hidden_units, res_dropout=0., embed_reg=1e-4):
        """
        Deep&Crossing
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param res_dropout: A scalar. Dropout of resnet.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Deep_Crossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding layers
        embed_dim = sum([feat['embed_dim']
                         for feat in self.sparse_feature_columns])
        # the dimension of stack layer
        dim_stack = len(self.dense_feature_columns) + embed_dim
        self.res_network = [Residual_Units(
            unit, dim_stack) for unit in hidden_units]
        self.res_dropout = Dropout(res_dropout)
        self.dense = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        stack = tf.concat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs


class DIN(Model):
    def __init__(self, user_num, item_num, cate_num, cate_list, hidden_units):
        """
        :param user_num: 用户数量
        :param item_num: 物品数量
        :param cate_num: 物品种类数量
        :param cate_list: 物品种类列表
        :param hidden_units: 隐藏层单元
        """
        super(DIN, self).__init__()
        self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int32)
        self.hidden_units = hidden_units
        # self.user_embed = layers.Embedding(
        #     input_dim=user_num, output_dim=hidden_units, embeddings_initializer='random_uniform',
        #     embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='user_embed')
        self.item_embed = layers.Embedding(
            input_dim=item_num, output_dim=self.hidden_units, embeddings_initializer='random_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='item_embed')
        self.cate_embed = layers.Embedding(
            input_dim=cate_num, output_dim=self.hidden_units, embeddings_initializer='random_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='cate_embed'
        )
        self.dense = layers.Dense(self.hidden_units)
        self.bn1 = layers.BatchNormalization()
        self.concat = layers.Concatenate(axis=-1)
        self.att_dense1 = layers.Dense(80, activation='sigmoid')
        self.att_dense2 = layers.Dense(40, activation='sigmoid')
        self.att_dense3 = layers.Dense(1)
        self.bn2 = layers.BatchNormalization()
        self.concat2 = layers.Concatenate(axis=-1)
        self.dense1 = layers.Dense(80, activation='sigmoid')
        self.activation1 = layers.PReLU()
        # self.activation1 = Dice()
        self.dense2 = layers.Dense(40, activation='sigmoid')
        self.activation2 = layers.PReLU()
        # self.activation2 = Dice()
        self.dense3 = layers.Dense(1, activation=None)

    def call(self, inputs):
        user, item, hist, sl = inputs[0], tf.squeeze(
            inputs[1], axis=1), inputs[2], tf.squeeze(inputs[3], axis=1)
        # user_embed = self.u_embed(user)
        item_embed = self.concat_embed(item)
        hist_embed = self.concat_embed(hist)
        # 经过attention的物品embedding
        hist_att_embed = self.attention(item_embed, hist_embed, sl)
        hist_att_embed = self.bn1(hist_att_embed)
        hist_att_embed = tf.reshape(
            hist_att_embed, [-1, self.hidden_units * 2])
        u_embed = self.dense(hist_att_embed)
        item_embed = tf.reshape(item_embed, [-1, item_embed.shape[-1]])
        # 联合用户行为embedding、候选物品embedding、【用户属性、上下文内容特征】
        embed = self.concat2([u_embed, item_embed])
        x = self.bn2(embed)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        outputs = tf.nn.sigmoid(x)
        return outputs

    def summary(self):
        user = Input(shape=(1,), dtype=tf.int32)
        item = Input(shape=(1,), dtype=tf.int32)
        sl = Input(shape=(1,), dtype=tf.int32)
        hist = Input(shape=(431,), dtype=tf.int32)
        Model(inputs=[user, item, hist, sl], outputs=self.call(
            [user, item, hist, sl])).summary()

    def concat_embed(self, item):
        """
        拼接物品embedding和物品种类embedding
        :param item: 物品id
        :return: 拼接后的embedding
        """
        # cate = tf.transpose(tf.gather_nd(self.cate_list, [item]))
        cate = tf.gather(self.cate_list, item)
        cate = tf.squeeze(cate, axis=1) if cate.shape[-1] == 1 else cate
        item_embed = self.item_embed(item)
        item_cate_embed = self.cate_embed(cate)
        embed = self.concat([item_embed, item_cate_embed])
        return embed

    def attention(self, queries, keys, keys_length):
        """
        activation unit
        :param queries: item embedding
        :param keys: hist embedding
        :param keys_length: the number of hist_embed
        :return:
        """
        # 候选物品的隐藏向量维度，hidden_unit * 2
        queries_hidden_units = queries.shape[-1]
        # 每个历史记录的物品embed都需要与候选物品的embed拼接，故候选物品embed重复keys.shape[1]次
        # keys.shape[1]为最大的序列长度，即431，为了方便矩阵计算
        # [None, 431 * hidden_unit * 2]
        queries = tf.tile(queries, [1, keys.shape[1]])
        # 重塑候选物品embed的shape
        # [None, 431, hidden_unit * 2]
        queries = tf.reshape(
            queries, [-1, keys.shape[1], queries_hidden_units])
        # 拼接候选物品embed与hist物品embed
        # [None, 431, hidden * 2 * 4]
        embed = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)
        # 全连接, 得到权重W
        d_layer_1 = self.att_dense1(embed)
        d_layer_2 = self.att_dense2(d_layer_1)
        # [None, 431, 1]
        d_layer_3 = self.att_dense3(d_layer_2)
        # 重塑输出权重类型, 每个hist物品embed有对应权重值
        # [None, 1, 431]
        outputs = tf.reshape(d_layer_3, [-1, 1, keys.shape[1]])

        # Mask
        # 此处将为历史记录的物品embed令为True
        # [None, 431]
        key_masks = tf.sequence_mask(keys_length, keys.shape[1])
        # 增添维度
        # [None, 1, 431]
        key_masks = tf.expand_dims(key_masks, 1)
        # 填充矩阵
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # 构造输出矩阵，其实就是为了实现【sum pooling】。True即为原outputs的值，False为上述填充值，为很小的值，softmax后接近0
        # [None, 1, 431] ----> 每个历史浏览物品的权重
        outputs = tf.where(key_masks, outputs, paddings)
        # Scale，keys.shape[-1]为hist_embed的隐藏单元数
        outputs = outputs / (keys.shape[-1] ** 0.5)
        # Activation，归一化
        outputs = tf.nn.softmax(outputs)
        # 对hist_embed进行加权
        # [None, 1, 431] * [None, 431, hidden_unit * 2] = [None, 1, hidden_unit * 2]
        outputs = tf.matmul(outputs, keys)
        return outputs
