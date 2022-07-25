from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from TargetEncoder import *
import numpy as np
import pandas as pd
from itertools import repeat


def load_csv():
    # data = pd.read_csv('data/source/CSV_source_new.csv', header=0, encoding='ANSI',low_memory=False)
    # 去除无用特征
    # data.drop(['ts','uid','id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'trans_id', 'proto','rtt', 'qclass','qtype','qtype_name','qclass_name','service','local_orig','local_resp','tunnel_parents','rcode_name','AA', 'TC','rejected'],
    #         inplace=True, axis=1)

    # ohe = OneHotEncoder()
    # Encoder_features = ['State', 'Registrant_Name', 'Country.1', 'Creation_Date_Time','Domain_Name'
    #     ,'Organization','longest_word','1gram','Domain_Age','tld','Emails','typos','3gram','char_distribution','2gram','Registrar','sld']
    # data=data.fillna(0)
    # data_transform=ohe.fit_transform(data[Encoder_features]).toarray()
    # feature_names=ohe.get_feature_names()
    # data_transform=pd.DataFrame(data_transform,columns=feature_names)
    # data = data.join(data_transform)
    # data.drop(Encoder_features, inplace=True, axis=1)
    # data.to_csv('data/source/features2.csv', index=False)
    #获得列名
    # columns_name = data.columns.tolist()
    # #打乱数据顺序
    # # data = shuffle(data)
    # # #分出训练集和测试集
    # # data_train = data[:23073]
    # # data_test = data[23073:]
    # # data_train.columns = columns_name
    # # data_test.colums = columns_name

    data = pd.read_csv('data/source/features2.csv', header=0, encoding='ANSI', low_memory=False)
    Label = np.array(data['label'])
    Y = Label
    data = data.drop(['label'], axis=1)
    # # 判断nan和无穷大，转化为0
    X = data.astype(float)
    np.nan_to_num(X)
    X[np.isnan(X)] = 0
    X_inf = np.isinf(X)
    X[X_inf] = 0
    X = X.values
    #
    # # 归一化
    scaler = preprocessing.MinMaxScaler()
    MinMax_Scaler = scaler.fit(X)
    X = MinMax_Scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    # RF = RandomForestClassifier(n_estimators=300, n_jobs=4, max_features='auto', random_state=12, bootstrap=True)
    # RF.fit(X_train, Y_train)  # 拟合模型
    # Y_pre = RF.predict(X_test)
    # print(classification_report(Y_test, Y_pre, digits=5))
    # importances = RF.feature_importances_
    # feat_labels=data.columns
    # indices = np.argsort(importances)[::-1]
    # for f in range(X_train.shape[1]):
    #     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    # knn
    # kn = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
    # kn.fit(X_train, Y_train)
    # Y_pre = kn.predict(X_test)
    # print(classification_report(Y_test, Y_pre, digits=5))
    # # 逻辑回归
    log_reg=LogisticRegression(C=1.0, tol=1e-6,solver='lbfgs')        #1-1定义一种二分类算法
    ovr=OneVsRestClassifier(log_reg)    #1-2进行多分类转换
    ovr.fit(X_train,Y_train)
    Y_pre = ovr.predict(X_test)#1-3进行数据训练与预测
    print(classification_report(Y_test, Y_pre, digits=5))
    # 支持向量机
    # model = SVC(decision_function_shape="ovr")
    # model.fit(X_train,Y_train)
    # Y_pre = model.predict(X_test)
    # print(classification_report(Y_test, Y_pre, digits=5))
    # clf=GaussianNB()
    # ovr=OneVsRestClassifier(clf)    #进行多分类转换
    # ovr.fit(X_train,Y_train)
    # Y_pre = ovr.predict(X_test)
    # print(classification_report(Y_test, Y_pre, digits=5))
    # 多层感知机
    # model = MLPClassifier(hidden_layer_sizes=(30,20,10),activation='relu', solver='adam', alpha=0.001)
    # model.fit(X_train,Y_train)
    # Y_pre = model.predict(X_test)
    # print(classification_report(Y_test, Y_pre, digits=5))
    # # 将矩阵转成csv
    # np.savetxt('./data/domain_train.csv', domain_train, delimiter = ',',fmt='%s')
    # np.savetxt('./data/domain_test.csv', domain_test, delimiter=',',fmt='%s')
    # np.savetxt('./data/train.csv', X_train, delimiter=',')
    # np.savetxt('./data/test.csv', X_test, delimiter=',')
    # # np.savetxt('./data/word+train.csv', X_train, delimiter = ',')
    # # np.savetxt('./data/word+test.csv', X_test, delimiter=',')
    # np.savetxt('./data/Y_train.csv', Y_train, delimiter=',',fmt = '%d')
    # np.savetxt('./data/Y_test.csv', Y_test, delimiter=',',fmt = '%d')

if __name__ == '__main__':
    load_csv()
