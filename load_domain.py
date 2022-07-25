import numpy as np

def loadDomain(data):
    data=data['query']
    # define documents
    charList = {}
    #字符序列要从1开始,0是填充字符
    i=1
    with open('./data/charList.txt') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').strip(' ')
            if temp != '':
                charList[temp] = i
                i += 1
    x_data_sum = []
    for line in data:

        x_data = []
        for char in line:
            try:
                x_data.append(charList[char])
            except:
                print('unexpected char' + ' : ' + char)
                x_data.append(0)
        x_data_sum.append(x_data)
    # for line in data:
    #     x_data = []
    #     for i in range(len(line)):
    #         try:
    #             x_data.append(line[i])
    #         except:
    #             print('unexpected char' + ' : ' + line[i])
    #             x_data.append(0)
    #     x_data_sum.append(x_data)
    x_data_sum = np.array(x_data_sum)

    return x_data_sum