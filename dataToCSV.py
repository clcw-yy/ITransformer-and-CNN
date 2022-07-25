from TargetEncoder import *
from load_data import loadData
from Wordfeature import *
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import numpy as np
def load_csv():
    data = pd.read_csv('./data/csv/new_all.csv', header=0)
    # 去除无用特征
    data.drop(['ts','uid','id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'trans_id', 'proto','rtt', 'qclass','qtype','qtype_name','qclass_name','service','local_orig','local_resp','tunnel_parents','rcode_name','AA', 'TC','rejected'],
            inplace=True, axis=1)

    # # 添加词汇特征
    data = Word_feature(data)
    data.drop(['query'], inplace=True, axis=1)
    # 处理answers字段和TTLs字段
    data['answers'] = data['answers'].replace('-', '')
    answer_list = []
    for row in data['answers']:
        arr = row.split(',')
        for key in arr:
            if key == '':
                answer_list.append(0)
                break
            else:
                answer_list.append(len(arr))
                break
    data['answers_target'] = answer_list
    data.drop(['answers'], inplace=True, axis=1)

    data['TTLs'] = data['TTLs'].replace('-', '')
    ttl_list = []
    for row in data['TTLs']:
        sum = 0
        arr = row.split(',')
        for key in arr:
            if key == '':
                break
            else:
                sum = sum + float(key)
        ttl_list.append(sum / len(arr))
    data['TTLs_target'] = ttl_list
    data.drop(['TTLs'], inplace=True, axis=1)

    #独热编码
    ohe=OneHotEncoder()
    Encoder_features = ['RD', 'RA', 'conn_state', 'history']
    data_transform=ohe.fit_transform(data[Encoder_features]).toarray()
    feature_names=ohe.get_feature_names()
    data_transform=pd.DataFrame(data_transform,columns=feature_names)
    data = data.join(data_transform)
    data.drop(Encoder_features, inplace=True, axis=1)

    #获得列名
    columns_name = data.columns.tolist()
    #打乱数据顺序
    # data = shuffle(data)
    # #分出训练集和测试集
    # data_train = data[:23073]
    # data_test = data[23073:]
    # data_train.columns = columns_name
    # data_test.colums = columns_name

    # 判断nan和无穷大，转化为0
    X = data.replace('-', 0)
    X = X.astype(float)
    np.nan_to_num(X)
    X[np.isnan(X)] = 0
    X_inf = np.isinf(X)
    X[X_inf] = 0
    X = X.values

    # 归一化
    scaler = preprocessing.MinMaxScaler()
    MinMax_Scaler = scaler.fit(X)
    X = MinMax_Scaler.transform(X)

    data=pd.DataFrame(X,columns=columns_name)
    data.to_csv("./data/new_word_all.csv", index=False)
    # domain_train = data_train['query']
    # domain_test = data_test['query']
    #
    # data_train.drop(['query'],inplace = True, axis = 1)
    # data_test.drop(['query'], inplace=True, axis=1)
    # X_train,Y_train,X_test,Y_test=loadData(data_train,data_test)
    #
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
