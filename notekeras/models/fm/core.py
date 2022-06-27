import tensorflow as tf
from notekeras.component import Component
from notekeras.models.fm.model import DNN, FactorizationMachine
from tensorflow.keras.layers import Dense


class DeepFM(Component):
    def __init__(self,
                 output_dim=1,
                 activation='sigmoid',
                 dnn_units=(200, 200, 200),
                 dnn_dropout=0.,
                 dnn_activation='relu',
                 fm_factor_dim=10,
                 fm_output_dim=1,
                 fm_w_reg=1e-4,
                 fm_v_reg=1e-4,
                 *args, **kwargs):
        """
        DeepFM
        :param feature_columns: A list. a list containing dense and sparse column feature information.
        :param fm_factor_dim: A scalar. fm's latent vector number.
        :param dnn_units: A list. A list of dnn hidden units.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param activation: A string. Activation function of dnn.
        :param fm_w_reg: A scalar. The regularizer of w in fm.
        :param fm_v_reg: A scalar. The regularizer of v in fm.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DeepFM, self).__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.activation = activation

        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation

        self.fm_v_reg = fm_v_reg
        self.fm_w_reg = fm_w_reg
        self.fm_factor_dim = fm_factor_dim
        self.fm_output_dim = fm_output_dim

        self.fm = self.dense = self.w1 = self.w2 = self.bias = self.dnn = None

    def build(self, input_shape):
        self.fm = FactorizationMachine(output_dim=self.fm_output_dim,
                                       factor_dim=self.fm_factor_dim,
                                       kernal_reg=self.fm_v_reg,
                                       weight_reg=self.fm_w_reg,
                                       name='FM')

        self.dnn = DNN(self.dnn_units, self.dnn_activation, self.dnn_dropout)
        self.dense = Dense(self.output_dim, activation=self.activation)

    def call(self, inputs, **kwargs):
        wide_outputs = self.fm(inputs)
        deep_outputs = self.dnn(inputs)
        outputs = tf.keras.layers.Concatenate()([wide_outputs, deep_outputs])
        outputs = self.dense(outputs)
        return outputs
