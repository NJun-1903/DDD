from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

# Đăng ký lớp AttLayer với Keras để có thể tải mô hình
@register_keras_serializable()
class AttLayer(layers.Layer):
    """
    Attention layer for use with Keras
    """
    def __init__(self, attention_dim, **kwargs):
        """
        Initialize a new attention layer
        Args:
            attention_dim (int): Internal dimension of this layer
        """
        super(AttLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        """
        Initialize the internal matrices of the layer given input shape
        Args:
            input_shape (tuple): Shape of input layer (3D)
        """
        assert len(input_shape) == 3
        self.w = self.add_weight(shape=(input_shape[-1], self.attention_dim), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.attention_dim,), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(self.attention_dim, 1), initializer='random_normal', trainable=True)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        """
        Calculate the output of this layer
        Args:
            x: Input tensor
            mask (default None):

        Returns:
            output tensor
        """
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.w), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        # output = K.max(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]