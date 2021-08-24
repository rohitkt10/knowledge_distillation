import tensorflow as tf
from tensorflow import keras as tfk


__all__ = ['get_model']

def get_model(
    input_shape=(1000, 4),
    num_classes=919,
    activation=tf.nn.relu,
    first_activation=None,
    kernel_initializer="glorot_normal",
    logits_only=False,
    name="deepatt",
	num_heads=4,
	num_dimensions=400,
			):
	assert num_dimensions%num_heads==0, 'The dimensionality of the MultiHeadAttention layer \
										must be divisible by the number of attention heads.'
	key_dim = int(num_dimensions/num_heads)

	# layer 0
	x = tfk.layers.Input(input_shape, name='input')

	# 1st conv layer
	y = tf.keras.layers.Conv1D(
						filters=1024,
						kernel_size=30,
						strides=1,
						padding="same",
						activation=first_activation or "relu",
					)(x)
	y = tfk.layers.MaxPool1D(pool_size=10,)(y)
	y = tfk.layers.Dropout(0.2)(y)

	# bi directional LSTM layer
	lstm_kwargs = {'units':512, 'return_sequences':True, 'return_state':True}
	lstm_forward = tfk.layers.LSTM(**lstm_kwargs)
	lstm_backward = tfk.layers.LSTM(go_backwards=True, **lstm_kwargs)
	bidlstm = tfk.layers.Bidirectional(
						lstm_forward, backward_layer=lstm_backward,
									  )
	y, *res = bidlstm(y)
	print(y.shape)

	# Multihead Attention
	mha = tfk.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
	y, *res = mha(y,y,y, return_attention_scores=True)
	y = tfk.layers.Dropout(0.2)(y)

	# point wise feed foward network
	y = tfk.layers.Dense(100, activation='relu',)(y)  ## (batch, seq_len, 100)
	y =	tfk.layers.Dense(num_classes, activation='linear')(y) ## (batch, seq_len, num classes)
	y = tfk.layers.GlobalMaxPooling1D(data_format='channels_last')(y)

	if not logits_only:
		y = tfk.layers.Activation('sigmoid', name='probabilities')(y)

	# final model
	model = tfk.Model(x, y, name=name)
	return model

def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, None]  ## array of positions ; shape = (L, 1)
    I = np.arange(d_model)[None, :]  ## array of feature indices ; shape = (1, d_model)
    angle_rads = positions / np.power(10000, 2*(I//2) / np.float32(d_model))  ## (L, d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) ## even indexed features with sine encoding
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) ## odd indexed features with cosine encoding
    pos_encoding = angle_rads[None, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class DeepAtt(tf.keras.Model):
    def __init__(self, first_activation, **unused):
        super(DeepAtt, self).__init__(name="deepatt")
        self.conv_1 = tf.keras.layers.Conv1D(
            filters=1024,
            kernel_size=30,
            strides=1,
            padding="valid",
            activation=first_activation or "relu",
        )

        self.pool_1 = tf.keras.layers.MaxPool1D(
            pool_size=15, strides=15, padding="valid"
        )

        self.dropout_1 = tf.keras.layers.Dropout(0.2)

        self.bidirectional_rnn = _BidLSTM(512)

        self.category_encoding = tf.eye(919)[tf.newaxis, :, :]

        self.multi_head_attention = _MultiHeadAttention(400, 4)

        self.dropout_2 = tf.keras.layers.Dropout(0.2)

        # Note: Point-wise-dense == Category Dense (weight-share).
        self.point_wise_dense_1 = tf.keras.layers.Dense(units=100, activation="relu")

        self.point_wise_dense_2 = tf.keras.layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of DeepAttention model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        batch_size = tf.shape(inputs)[0]

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 971, 1024]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 971, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training=training)

        # Bidirectional RNN Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask)

        # Category Multi-head Attention Layer 1
        # Input Tensor Shape: v.shape = [batch_size, 64, 1024]
        #                     k.shape = [batch_size, 64, 1024]
        #                     q.shape = [batch_size, 919, 919]
        # Output Tensor Shape: temp.shape = [batch_size, 919, 400]
        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])
        temp, _ = self.multi_head_attention(query, k=temp, v=temp)

        # Dropout Layer 2
        temp = self.dropout_2(temp, training=training)

        # Category Dense Layer 1 (weight-share)
        # Input Tensor Shape: [batch_size, 919, 400]
        # Output Tensor Shape: [batch_size, 919, 100]
        temp = self.point_wise_dense_1(temp)

        # Category Dense Layer 2 (weight-share)
        # Input Tensor Shape: [batch_size, 919, 100]
        # Output Tensor Shape: [batch_size, 919, 1]
        output = self.point_wise_dense_2(temp)

        output = tf.reshape(output, [-1, 919])

        return output

class _BidLSTM(tf.keras.layers.Layer):
    """Bidirectional LSTM Layer.
    Reference:
        - [LSTM](https://arxiv.org/abs/1402.1128)
    https://github.com/jiawei6636/Bioinfor-DeepATT/blob/651b0dc722fcf2407ef88b8c84587ca92ec1bdc1/model/layers/bidirection_rnn.py
    """

    def __init__(self, units=100):
        """
        Initialize the BidLSTM layer.
        :param units: num of hidden units.
        """
        super(_BidLSTM, self).__init__()
        forward_layer = tf.keras.layers.LSTM(
            units=units, return_sequences=True, return_state=True
        )
        backward_layer = tf.keras.layers.LSTM(
            units=units, return_sequences=True, return_state=True, go_backwards=True
        )
        self.bidirectional_rnn = tf.keras.layers.Bidirectional(
            layer=forward_layer, backward_layer=backward_layer
        )

    def build(self, input_shape):
        super(_BidLSTM, self).build(input_shape=input_shape)

    def call(self, inputs, mask=None, **kwargs):
        """
        Call function of BidLSTM layer.
        :param inputs: shape = (batch_size, time_steps, channel)
        :param mask: shape = (batch_size, time_steps)
        :param kwargs: None.
        :return: (sequence_output, state_output).
                  sequence_output shape is (batch_size, time_steps, units x 2),
                  state_output shape is (batch_size, units x 2)
        """
        output = self.bidirectional_rnn(inputs, mask=mask)
        sequence_output = output[0]
        forward_state_output = output[1]
        backward_state_output = output[2]
        state_output = tf.keras.layers.concatenate(
            [forward_state_output, backward_state_output], axis=-1
        )
        return sequence_output, state_output

    @staticmethod
    def create_padding_mask(seq_len, max_len):
        """
        Create the padding mask matrix according to the seq_len and max_len.
        Set the value to 0 to mask the padding sequence.
        :param seq_len: the sequence length.
        :param max_len: the max length.
        :return: padding mask matrix. (shape = (batch_size, max_len))
        """
        mask_matrix = tf.sequence_mask(seq_len, maxlen=max_len)
        return mask_matrix


class _MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention Layer.
    Multi-head attention by q, k, v.
    Schematic:
        1. Linear layer and split to multi heads.
        2. Scaled dot-product attention.
        3. Concatenate the heads.
        4. Final linear layer.
    Reference:
        Multi-Head Attention](https://arxiv.org/abs/1706.03762
        https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb
    https://github.com/jiawei6636/Bioinfor-DeepATT/blob/651b0dc722fcf2407ef88b8c84587ca92ec1bdc1/model/layers/multihead_attention.py
    """

    def __init__(self, num_dimensions, num_heads):
        """
        Initialize the MultiHeadAttention layer.
        :param num_dimensions: the number of the dimensions of the layer.
        :param num_heads: the number of the heads of the layer.
        """
        super(_MultiHeadAttention, self).__init__()
        # The num_dimensions must be divisible by num_heads.
        assert num_dimensions % num_heads == 0

        self.num_dimensions = num_dimensions
        self.num_heads = num_heads
        self.depth = self.num_dimensions // self.num_heads

        self.wq = tf.keras.layers.Dense(num_dimensions)
        self.wk = tf.keras.layers.Dense(num_dimensions)
        self.wv = tf.keras.layers.Dense(num_dimensions)
        self.dense = tf.keras.layers.Dense(num_dimensions)

    def build(self, input_shape):
        super(_MultiHeadAttention, self).build(input_shape=input_shape)

    def call(self, q, k=None, v=None, mask=None):
        """
        Call function of MultiHeadAttention.
        :param q: the query. shape = (batch_size, seq_len_q, None)
        :param k: the key. shape = (batch_size, seq_len_k, None)
        :param v: the value. shape = (batch_size, seq_len_v, None)
        :param mask: Padding_mask.shape = (batch_size, 1, 1, seq_len)
            / Lookahead_mask.shape = (seq_len, seq_len)
        :return: outputs and attention weights.
        """
        # 1\ Linear layer and split to multi heads.
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len_q, num_dimensions)
        k = self.wk(k)  # (batch_size, seq_len_k, num_dimensions)
        v = self.wv(v)  # (batch_size, seq_len_v, num_dimensions)
        q = self.split_heads(
            q, batch_size, self.num_heads, self.depth
        )  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size, self.num_heads, self.depth
        )  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size, self.num_heads, self.depth
        )  # (batch_size, num_heads, seq_len_v, depth)

        # 2\ Scaled dot-product attention.
        # attention_outputs.shape = (batch_size, num_heads, seq_len_q, depth_v)
        # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_outputs, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # 3\ Concatenate the heads.
        # temp.shape = (batch_size, seq_len_q, num_heads, depth)
        # concat_attention.shape = (batch_size, seq_len_q, num_dimensions)
        temp = tf.transpose(attention_outputs, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            temp, (batch_size, temp.shape[1], self.num_dimensions)
        )

        # 4\ Final linear layer.
        # output.shape = (batch_size, seq_len_q, num_dimensions)
        outputs = self.dense(concat_attention)

        return outputs, attention_weights

    @staticmethod
    def split_heads(x, batch_size, num_heads, depth):
        """
        Split the last dimension into (num_heads, depth).
        Then Transpose the result such that the shape is
            (batch_size, num_heads, seq_len, depth)
        :param x: shape = (batch_size, seq_len, num_dimensions)
        :param num_heads: batch size
        :param depth: depth
        :return: shape = (batch_size, num_heads, seq_len, depth)
        """
        temp = tf.reshape(x, (batch_size, x.shape[1], num_heads, depth))
        temp = tf.transpose(temp, perm=[0, 2, 1, 3])
        return temp

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """
        Calculate the attention weights.
        Schematic:
            1 Calculate the matmul_qk.
            2 Scale matmul_qk.
            3 Add the mask to the scaled tensor.
            4 Softmax and Weighted Summation.
        Note:
            1 q, k, v must have matching leading dimensions.
            2 q, k must have matching last dimensions. (depth_q = depth_v)
            3 k, v must have matching penultimate dimensions. (seq_len_k = seq_len_v)
            4 The mask has different shapes depending on its type (padding or look
                ahead),
               but it must be broadcastable for addition.
        :param q: query, shape = (batch_size, num_heads, seq_len_q, depth_q)
        :param k: key, shape = (batch_size, num_heads, seq_len_k, depth_k)
        :param v: value, shape = (batch_size, num_heads, seq_len_v, depth_v)
        :param mask: Float tensor with shape broadcastable to
            (batch_size, num_heads, seq_len_q, seq_len_k).
        :return: output, attention_weights
        """
        # 1\ Calculate the matmul_qk.
        matmul_qk = tf.matmul(
            q, k, transpose_b=True
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # 2\ Scale matmul_qk.
        d = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d)

        # 3\ Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # 4\ Softmax and Weighted Summation.
        # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        # attetion_outputs.shape = (batch_size, num_heads, seq_len_q, depth_v)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_outputs = tf.matmul(attention_weights, v)
        return attention_outputs, attention_weights

    @staticmethod
    def create_padding_mask(seq):
        """
        Create padding mask.
        Set 1 to mask the padding.
        :param seq: sequence. shape = (batch_size, seq_len)
        :return: mask matrix. shape = (batch_size, seq_len)
        """
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding to the attention logits.
        mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        return mask

    @staticmethod
    def create_look_ahead_mask(size):
        """
        Create look-ahead mask.
        Set 1 to mask the future information.
        :param size: size.
        :return: mask matrix. shape = (size, size)
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)


class _CategoryDense(tf.keras.layers.Layer):
    """CategoryDense
    https://github.com/jiawei6636/Bioinfor-DeepATT/blob/651b0dc722fcf2407ef88b8c84587ca92ec1bdc1/model/layers/category_dense.py
    """

    def __init__(
        self,
        units,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        """
        Initialize the Category Dense layer.
        :param units: num of hidden units.
        """
        super(_CategoryDense, self).__init__(**kwargs)
        self.units = units
        self.kernel = None
        self.bias = None
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        super(_CategoryDense, self).build(input_shape=input_shape)
        category = input_shape[1]
        input_channel = input_shape[2]
        output_channel = self.units
        kernel_shape = [1, category, input_channel, output_channel]
        bias_shape = [1, category, output_channel]
        self.kernel = self.add_weight(
            shape=kernel_shape,
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.bias = self.add_weight(
            shape=bias_shape,
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Call function of Category Dense layer.
        :param inputs: shape = (batch_size, Categories, channel)
        :return: shape = (batch_size, Categories, output_channel)
        """
        inputs = inputs[:, :, :, tf.newaxis]
        outputs = tf.reduce_sum(tf.multiply(inputs, self.kernel), axis=2)
        outputs = tf.add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)

        return outputs
