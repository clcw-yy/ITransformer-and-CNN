import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                 max_seq_len, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.n_layers = n_layers

        self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)
        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads,max_seq_len=max_seq_len,
                                          middle_units=middle_units,
                                          epsilon=epsilon, dropout_rate=dropout_rate,
                                          training=training)
                             for _ in range(n_layers)]

    def call(self, inputs, **kwargs):
        emb, mask = inputs
        emb=self.pos_embedding(emb)
        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb, mask)

        return emb


# 编码层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_seq_len,middle_units, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(num_heads)
        self.ffn = point_wise_feed_forward_network(d_model+max_seq_len,middle_units)
        # self.ffn = point_wise_feed_forward_network(d_model, middle_units)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.training = training

    def call(self, inputs, mask, **kwargs):
        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs, mask])
        att_output = self.dropout1(att_output, training=self.training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=self.training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# 层标准化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


# 前向网络
def point_wise_feed_forward_network(numUnits,middle_units):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(numUnits, activation='relu')])


# dot attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output


# 构造 multi head attention 层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

        self.dot_attention = scaled_dot_product_attention

    def split_heads(self, x, batch_size,depth):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads,depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]
        numUnits=q.get_shape().as_list()[-1]
        # 分头后的维度
        depth = numUnits // self.num_heads
        # 分头前的前向网络，获取q、k、v语义
        wq = tf.keras.layers.Dense(numUnits)  # (batch_size, seq_len, d_model)
        wk = tf.keras.layers.Dense(numUnits)
        wv = tf.keras.layers.Dense(numUnits)
        q=wq(q)
        k=wk(k)
        v=wv(v)

        # 分头
        q = self.split_heads(q, batch_size,depth)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size,depth)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size,depth)  # (batch_size, num_heads, seq_len_v, depth)
        # 通过缩放点积注意力层
        scaled_attention = self.dot_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth)

        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, numUnits))

        # 全连接层
        dense = tf.keras.layers.Dense(numUnits)
        output = dense(concat_attention)

        return output

# mask功能
def padding_mask(seq):
    # 获取为 0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 位置编码
# class PositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
#         self.sequence_len = sequence_len
#         self.embedding_dim = embedding_dim
#         super(PositionalEncoding, self).__init__(**kwargs)
#
#     def call(self, inputs):
#         if self.embedding_dim == None:
#             self.embedding_dim = int(inputs.shape[-1])
#
#         position_embedding = np.array([
#             [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
#             for pos in range(self.sequence_len)])
#
#         position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
#         position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
#
#         position_embedding = tf.cast(position_embedding, dtype=tf.float32)
#
#         return position_embedding + inputs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None,embedding_dim=None,**kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        embeddedPosition = []
        for step in range(self.sequence_len ):
            a = np.zeros(self.sequence_len )
            a[step] = 1
            embeddedPosition.append(a)
        embeddedPosition=np.array(embeddedPosition, dtype="float32")
        position_embedding =tf.reshape(
                tf.tile(embeddedPosition, (tf.shape(inputs)[0], 1)),
                (-1, self.sequence_len, self.sequence_len)
                )
        position_embedding = tf.concat([inputs, position_embedding], -1)
        return position_embedding

if __name__ == "__main__":
    n_layers = 2
    d_model = 128
    num_heads = 4
    middle_units = 256
    max_seq_len = 40

    samples = 10
    training = False

    encode_padding_mask_list = padding_mask(np.random.randint(0, 108, size=(samples, max_seq_len)))
    input_data = tf.random.uniform((samples, max_seq_len, d_model))

    sample_encoder = Encoder(n_layers, d_model, num_heads, middle_units, max_seq_len, training)
    sample_encoder_output = sample_encoder([input_data, encode_padding_mask_list])
    # sess = tf.compat.v1.Session()
    # with sess.as_default():
    #     print(type(sess.run(sample_encoder_output)))
    print(type(sample_encoder_output))
    print(type(tf.constant([1, 2, 3, 4, 5, 6])))
    sess = tf.compat.v1.InteractiveSession()  # 创建一个新的计算图
    sess.run(tf.compat.v1.global_variables_initializer())  # 初始化所有参数
    print(sess.run(sample_encoder_output))
