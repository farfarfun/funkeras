import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2


class FactorizationMachine(Layer):

    def __init__(self,
                 output_dim=1,
                 factor_dim=10,
                 name='FM',
                 activation=None,
                 use_weight=True,
                 use_bias=True,
                 kernel_reg=1e-4,
                 weight_reg=1e-4,
                 *args,
                 **kwargs):
        """
        Factorization Machines
        :param output_dim: 输出维度
        :param factor_dim: 隐含向量维度
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FactorizationMachine, self).__init__(name=name, *args, **kwargs)
        self.output_dim = output_dim
        self.factor_dim = factor_dim
        self.activate = activation
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.kernel_reg = kernel_reg
        self.weight_reg = weight_reg
        self.weight = self.bias = self.kernel = None
        self.activate_layer = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      regularizer=l2(self.kernel_reg),
                                      trainable=True)
        if self.use_weight:
            self.weight = self.add_weight(name='weight',
                                          shape=(
                                              input_shape[1], self.output_dim),
                                          initializer='glorot_uniform',
                                          regularizer=l2(self.weight_reg),
                                          trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)

        self.activate_layer = activations.get(self.activate)
        super(FactorizationMachine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        xv_a = K.square(K.dot(inputs, self.kernel))
        xv_b = K.dot(K.square(inputs), K.square(self.kernel))
        p = 0.5 * K.sum(xv_a - xv_b, 1)
        xv = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        res = xv
        if self.use_weight:
            res = res + K.dot(inputs, self.weight)
        if self.use_bias:
            res = res + self.bias

        output = K.reshape(res, (-1, self.output_dim))

        if self.activate_layer is not None:
            output = self.activate_layer(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "factor_dim": self.factor_dim,
            "name": self.name,
            "kernel_reg": self.kernel_reg,
            "weight_reg": self.weight_reg,
            "use_weight": self.use_weight,
            "use_bias": self.use_bias,
            "activation": self.activate
        }
        config.update(super(FactorizationMachine, self).get_config())
        return config


FM = FactorizationMachine


class FFM(Layer):
    def __init__(self,
                 field_num=3,
                 factor_dim=10,
                 use_weight=True,
                 use_bias=True,
                 weight_reg=1e-3,
                 kernel_reg=5e-2,
                 activation=None):
        """
        :param factor_dim:
        :param field_num:
        :param weight_reg:
        :param kernel_reg:
        """
        super(FFM, self).__init__()
        self.field_num = field_num
        self.factor_dim = factor_dim

        self.use_weight = use_weight
        self.use_bias = use_bias

        self.weight_reg = weight_reg
        self.kernel_reg = kernel_reg
        self.activation = activation

        self.kernel = self.weight = self.bias = self.activate_layer = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          input_shape[1], self.field_num, self.factor_dim),
                                      initializer=tf.random_normal_initializer(),
                                      regularizer=l2(self.kernel_reg),
                                      trainable=True)

        self.weight = self.add_weight(name='weight', shape=(input_shape[1], 1),
                                      initializer=tf.random_normal_initializer(),
                                      regularizer=l2(self.weight_reg),
                                      trainable=True)

        self.bias = self.add_weight(name='bias', shape=(
            1,), initializer=tf.zeros_initializer(), trainable=True)
        self.activate_layer = activations.get(self.activation)

    def call(self, inputs, **kwargs):
        field_f = tf.tensordot(inputs, self.kernel, axes=[1, 0])

        output = 0
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                output += tf.reduce_sum(tf.multiply(
                    field_f[:, i], field_f[:, j]), axis=1, keepdims=True)

        if self.use_weight:
            output += tf.matmul(inputs, self.weight)

        if self.use_bias:
            output += self.bias

        if self.activate_layer:
            output = self.activate_layer(output)

        return output

    def get_config(self):
        config = {
            "name": self.name,
            "field_num": self.field_num,
            "factor_dim": self.factor_dim,
            "use_weight": self.use_weight,
            "use_bias": self.use_bias,
            "weight_reg": self.weight_reg,
            "kernel_reg": self.kernel_reg,
            "activation": self.activation
        }
        config.update(super(FFM, self).get_config())
        return config
