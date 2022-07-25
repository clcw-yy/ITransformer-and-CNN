import time
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.layers import Embedding,LSTM, Dense, Dropout,Flatten, Conv1D, MaxPool1D
from transformer import Encoder,padding_mask,PositionalEncoding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from load_domain import loadDomain


if __name__ == "__main__":
    df = pd.read_csv("./data/source/features2.csv")
    X = df.drop(['label'], axis=1)
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, shuffle=True,
                                                        random_state=2)
    # knn
    kn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    kn.fit(X_train, Y_train)
    Y_pre = kn.predict(X_test)
    print(classification_report(Y_test, Y_pre, digits=5))
    # mlp
    model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
    model.fit(X_train,Y_train)
    Y_pre = model.predict(X_test)
    print(classification_report(Y_test, Y_pre, digits=5))


    model = SVC(decision_function_shape="ovr")
    model.fit(X_train,Y_train)
    Y_pre = model.predict(X_test)
    print(classification_report(Y_test, Y_pre, digits=5))
