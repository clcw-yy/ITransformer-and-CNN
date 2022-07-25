from sklearn import preprocessing
from TargetEncoder import *


def loadData(train_data,test_data):
    train_data,Y_train=encoding(train_data)
    test_data,Y_test=encoding(test_data)
    # 归一化
    scaler = preprocessing.MinMaxScaler()
    MinMax_Scaler = scaler.fit(train_data)
    X_train = MinMax_Scaler.transform(train_data)
    X_test=MinMax_Scaler.transform(test_data)

    return X_train,Y_train,X_test,Y_test

def encoding(data):

    # 处理csv
    # TargetEncoder_features = ['AA', 'TC', 'RD', 'RA','conn_state', 'history','rejected']
    # target_enc = TargetEncoder(cols=TargetEncoder_features)
    # data['label'][np.isnan(data['label'])] = 0
    # target_enc.fit(data[TargetEncoder_features], data['label'])
    # data = data.join(target_enc.transform(data[TargetEncoder_features]).add_suffix('_target'))
    # data.drop(TargetEncoder_features, inplace=True, axis=1)

    # 判断nan和无穷大，转化为0
    X = data.replace('-', 0)
    X = X.astype(float)
    np.nan_to_num(X)
    X[np.isnan(X)] = 0
    X_inf = np.isinf(X)
    X[X_inf] = 0
    Label = np.array(X['label'])
    Y = Label
    X = X.drop(['label'], axis=1)
    X = X.values

    return X, Y