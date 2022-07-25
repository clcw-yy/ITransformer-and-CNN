import time

from keras import Input, Model
from keras_preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from load_domain import loadDomain
import numpy as np
from TargetEncoder import *
from keras.layers import Embedding,LSTM, Dense, Dropout,Flatten, Conv1D, MaxPool1D
from sklearn.metrics import classification_report


if __name__ == "__main__":
    X_train = pd.read_csv('data/train.csv', header=None)
    Y_train = pd.read_csv('data/Y_train.csv', header=None)
    X_test = pd.read_csv('data/test.csv', header=None)
    Y_test = pd.read_csv('data/Y_test.csv', header=None)
    domain_train = pd.read_csv('data/domain_train.csv',header=0)
    domain_test = pd.read_csv('data/domain_test.csv',header=0)

    domain_train=loadDomain(domain_train)
    domain_test=loadDomain(domain_test)


    domain_train = sequence.pad_sequences(domain_train, maxlen=40)
    domain_test=sequence.pad_sequences(domain_test, maxlen=40)

    inputs_A = Input(shape=(40,), dtype='int32')
    # 构建模型
    embeddings = Embedding(95,128)(inputs_A)
    out_seq = LSTM(128)(embeddings)
    out_seq = Dropout(0.3)(out_seq)

    outputs_A = Dense(4, activation='softmax')(out_seq)
    opt = Adam(lr=0.0001, decay=0.00001)
    model = Model(inputs=inputs_A, outputs=outputs_A)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #整合模型
    print('Train...')
    start = time.time()

    model.fit(domain_train, Y_train,
                        batch_size=32,
                        epochs=80,
                        validation_split=0.2)
    predict_test = model.predict(domain_test)
    predict = np.argmax(predict_test, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引
    print(classification_report(Y_test, predict, digits=5))