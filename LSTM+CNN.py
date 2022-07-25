import time
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import roc_auc_score, auc
from sklearn import metrics
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.layers import Embedding,LSTM, Dense, Dropout,Flatten, Conv1D, MaxPool1D
from transformer import Encoder,padding_mask,PositionalEncoding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from load_domain import loadDomain

#配置参数
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 128

    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 40  # 取了所有序列长度的均值
    batchSize = 32
    max_features=95
    num_features=37
    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目

    training = TrainingConfig()

    model = ModelConfig()


# 输出batch数据集

class LSTM_CNN(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                 max_seq_len,max_features,embeddingSize,**kwargs):
        super(LSTM_CNN, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.d_model = d_model
        self.max_seq_len=max_seq_len
        self.num_heads=num_heads
        self.middle_units=middle_units
        self.max_features=max_features
        self.embeddingSize=embeddingSize

    def call(self, inputs, **kwargs):
        inputs_A,inputs_B = inputs
        embeddings = Embedding(self.max_features,self.embeddingSize)(inputs_A)
        mask_inputs = padding_mask(inputs_A)
        out_seq = LSTM(128)(embeddings)
        out_seq = Dropout(0.3)(out_seq)
        outputs_A = Dense(64, activation='softmax')(out_seq)
        # outputs_A=out_seq

        # 卷积层
        c1 = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs_B)
        # 池化层
        m1 = MaxPool1D(pool_size=3, strides=3)(c1)
        # 卷积层
        c2 = Conv1D(filters=32, kernel_size=2, activation='relu')(m1)
        # 池化层
        m2 = MaxPool1D(pool_size=3, strides=3)(c2)
        d1 = Dropout(0.3)(m2)
        f1 = Flatten()(d1)
        # 全连接层
        # outputs_B = Dense(64, activation='softmax')(f1)
        # outputs=outputs_B
        # outputs = concatenate([outputs_A, outputs_B],1)
        # outputs = Dense(64, activation='relu')(outputs)
        outputs=Dense(4,activation='softmax')(f1)

        return outputs

if __name__ == "__main__":
    X_train = pd.read_csv('data/train.csv', header=None)
    Y_train = pd.read_csv('data/Y_train.csv', header=None)
    X_test = pd.read_csv('data/test.csv', header=None)
    Y_test = pd.read_csv('data/Y_test.csv', header=None)
    domain_train = pd.read_csv('data/domain_train.csv',header=0)
    domain_test = pd.read_csv('data/domain_test.csv',header=0)

    domain_train=loadDomain(domain_train)
    domain_test=loadDomain(domain_test)

    # 1. 实例化配置参数对象
    config = Config()
    X_train = X_train.values.reshape(len(X_train), config.num_features, 1)
    X_test=X_test.values.reshape(len(X_test),config.num_features, 1)
    domain_train = sequence.pad_sequences(domain_train, maxlen=config.sequenceLength)
    domain_test=sequence.pad_sequences(domain_test, maxlen=config.sequenceLength)
    # 构建模型LSTM
    inputs_A = Input(shape=(config.sequenceLength,), dtype='int32')
    embeddings = Embedding(config.max_features, config.model.embeddingSize)(inputs_A)
    mask_inputs = padding_mask(inputs_A)
    out_seq = LSTM(128)(embeddings)
    out_seq = Dropout(0.3)(out_seq)
    outputs_A = Dense(16, activation='softmax')(out_seq)
    #构建CNN
    inputs_B = Input(shape=(config.num_features,1), dtype='float32')
    # 卷积层
    c1 = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs_B)
    # 池化层
    m1 = MaxPool1D(pool_size=3, strides=3)(c1)
    # 卷积层
    c2 = Conv1D(filters=32, kernel_size=2, activation='relu')(m1)
    # 池化层
    m2 = MaxPool1D(pool_size=3, strides=3)(c2)
    d1 = Dropout(0.3)(m2)
    f1 = Flatten()(d1)
    # 全连接层
    outputs_B = Dense(16, activation='softmax')(f1)
    outputs = concatenate([outputs_A, outputs_B],1)
    outputs = Dense(4, activation='softmax')(outputs)

    model = Model(inputs=[inputs_A,inputs_B], outputs=outputs)
    print(model.summary())
    opt = Adam(lr=0.0001, decay=0.00001)
    loss = 'sparse_categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print('Train...')
    model.fit([domain_train,X_train], Y_train,
                        batch_size=config.batchSize,
                        epochs=100,
                        validation_split=0.2)
    predict_test = model.predict([domain_test, X_test])

    predict = np.argmax(predict_test, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引
    print(classification_report(Y_test, predict, digits=5))

