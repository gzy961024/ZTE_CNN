import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch


def data_tensor():
    # 读取原始数据集
    data = pd.read_csv('train.csv')

    # 数据清洗，填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    data_clean = imputer.fit_transform(data.iloc[:, :-1])
    data_clean = pd.DataFrame(data_clean, columns=data.columns[:-1])
    data_clean['label'] = data['label']

    # 特征标准化
    scaler = StandardScaler()
    data_clean.iloc[:, :-1] = scaler.fit_transform(data_clean.iloc[:, :-1])

    # 特征降维
    pca = PCA(n_components=100)
    data_reduced = pca.fit_transform(data_clean.iloc[:, :-1])
    data_reduced = pd.DataFrame(data_reduced)
    data_reduced['label'] = data_clean['label']
    # print(data_reduced)
    # 分离训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data_reduced.iloc[:, :-1], data_reduced['label'], test_size=0.2,
                                                        random_state=42)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    np_array = np.array(X_train.values)
    xtrain = torch.from_numpy(np_array)
    np_array = np.array(y_train.values)
    ytrain = torch.from_numpy(np_array)
    out_y = torch.zeros(ytrain.shape[0], 6)
    for num in range(out_y.shape[0]):
        out_y[num, ytrain[num].int().item()] = 1.0

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    np_array = np.array(X_test.values)
    testx = torch.from_numpy(np_array)
    np_array = np.array(y_test.values)
    testy = torch.from_numpy(np_array)
    # test_out_y = torch.zeros(testy.shape[0],6)
    # for x in range(test_out_y.shape[0]):
    #     test_out_y[x, testy[x].int().item()] = 1.0

    return xtrain, out_y, testx, testy

# print(X_train)

# import tensorflow as tf
# from sklearn.metrics import accuracy_score, recall_score

# 转化为LSTM需要的数据格式
# X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
# print(X_train_lstm.shape)
# # 创建LSTM模型
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# # 编译模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)
#
#
# # 预测测试集的节点类别
# y_pred = model.predict(X_test_lstm)
# y_pred1 = np.argmax(y_pred, axis=1)
# print(y_pred1)
# # 计算平均accuracy和平均recall
# accuracy = accuracy_score(y_test, y_pred1)
# recall = recall_score(y_test, y_pred1, average='macro')
#
# print('Average accuracy:', accuracy)
# print('Average recall:', recall)
# print("F1:",2*accuracy*recall/(recall+accuracy))
