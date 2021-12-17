#!/usr/bin/env python
# coding: utf-8

# In[65]:

# 导入必要的模块
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter


# 为日期转换做准备
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use('fivethirtyeight')

# 读取文件
df = pd.read_csv("中国石化.csv", encoding="gb2312")
df1 = pd.read_csv("中国石油.csv", encoding="gb2312")
df2 = pd.read_csv("中海油服.csv", encoding="gb2312")
df3 = pd.read_csv("泰山石油.csv", encoding="gb2312")
df4 = pd.read_csv("茂化实华：石油化工.csv", encoding="gb2312")

# In[67]:





# 去掉属性值为代码编号的列，并按日期从小到大进行排序
df = df.drop(columns=["ts_code"])
df = df.sort_values(by="trade_date", ascending=True)
df.set_index('trade_date')
df.to_csv(r'F:\600028.csv', index=0)

df1 = df1.drop(columns=["ts_code"])
df1 = df1.sort_values(by="trade_date", ascending=True)
df1.set_index('trade_date')
df1.to_csv(r'F:\60002801.csv', index=0)

df2 = df2.drop(columns=["ts_code"])
df2 = df2.sort_values(by="trade_date", ascending=True)
df2.set_index('trade_date')
df2.to_csv(r'F:\60002802.csv', index=0)

df3 = df3.drop(columns=["ts_code"])
d3f = df3.sort_values(by="trade_date", ascending=True)
df3.set_index('trade_date')
df3.to_csv(r'F:\60002803.csv', index=0)

df4 = df4.drop(columns=["ts_code"])
df4 = df4.sort_values(by="trade_date", ascending=True)
df4.set_index('trade_date')
df4.to_csv(r'F:\60002804.csv', index=0)

# In[68]:


# print(df)


# In[69]:


df = pd.read_csv(r'F:\600028.csv')

df['trade_date'] = df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['trade_date'], df['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

time_data = df.iloc[452:,0:1]
print(time_data)
time_data = np.array(time_data)

df['close'] = savgol_filter(df['close'], 5, 3, mode='nearest')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['trade_date'], df['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# In[70]:

df1 = pd.read_csv(r'F:\60002801.csv')

df1['trade_date'] = df1['trade_date'] = pd.to_datetime(df1['trade_date'], format='%Y%m%d')
print(df1)
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df1['trade_date'], df1['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

df1['close'] = savgol_filter(df1['close'], 5, 3, mode='nearest')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df1['trade_date'], df1['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# In[71]:


df2 = pd.read_csv(r'F:\60002802.csv')

df2['trade_date'] = df2['trade_date'] = pd.to_datetime(df2['trade_date'], format='%Y%m%d')
print(df2)
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df2['trade_date'], df2['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

df2['close'] = savgol_filter(df2['close'], 5, 3, mode='nearest')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df2['trade_date'], df2['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# In[72]:


df3 = pd.read_csv(r'F:\60002803.csv')

df3['trade_date'] = df3['trade_date'] = pd.to_datetime(df3['trade_date'], format='%Y%m%d')
print(df3)
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df3['trade_date'], df3['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

df3['close'] = savgol_filter(df3['close'], 5, 3, mode='nearest')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df3['trade_date'], df3['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# In[73]:


df4 = pd.read_csv(r'F:\60002804.csv')

df4['trade_date'] = df4['trade_date'] = pd.to_datetime(df4['trade_date'], format='%Y%m%d')
print(df4)
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df4['trade_date'], df4['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

df4['close'] = savgol_filter(df4['close'], 5, 3, mode='nearest')
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df4['trade_date'], df4['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# In[74]:


# em = pd.read_csv("新闻标题情感指数PLUS.csv", encoding="gb2312")

# em['日期'] = pd.to_datetime(em['日期'], format='%Y/%m/%d')
# print(em)
# link = pd.merge(left=df, left_on='trade_date',
                # right=em, right_on='日期')

# df = link
# print(df)

# In[75]:


# link1 = pd.merge(left=df1, left_on='trade_date',
                 # right=em, right_on='日期')

# df1 = link1
# print(df1)

# In[76]:


# link2 = pd.merge(left=df2, left_on='trade_date',
                 # right=em, right_on='日期')

# df2 = link2
# print(df2)

# In[77]:


# link3 = pd.merge(left=df3, left_on='trade_date',
                 # right=em, right_on='日期')

# df3 = link3
#print(df3)

# In[78]:


# link4 = pd.merge(left=df4, left_on='trade_date',
                 # right=em, right_on='日期')

# df4 = link4
# print(df4)

# In[79]:


data = df.filter(items=['close', 'scores'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)

training_data_len

# In[80]:


data1 = df1.filter(items=['close', 'scores'])
dataset1 = data1.values
training_data_len1 = math.ceil(len(dataset1) * .8)

training_data_len1

# In[81]:


data2 = df2.filter(items=['close', 'scores'])
dataset2 = data2.values
training_data_len2 = math.ceil(len(dataset2) * .8)

training_data_len2

# In[82]:


data3 = df3.filter(items=['close', 'scores'])
dataset3 = data3.values
training_data_len3 = math.ceil(len(dataset3) * .8)

training_data_len3

# In[83]:


data4 = df4.filter(items=['close', 'scores'])
dataset4 = data4.values
training_data_len4 = math.ceil(len(dataset4) * .8)

training_data_len4

# In[84]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# In[85]:


scaled_data1 = scaler.fit_transform(dataset1)

scaled_data1

# In[86]:


scaled_data2 = scaler.fit_transform(dataset2)

scaled_data2

# In[87]:


scaled_data3 = scaler.fit_transform(dataset3)

scaled_data3

# In[88]:


scaled_data4 = scaler.fit_transform(dataset4)

scaled_data4

# In[89]:


# 创建训练集
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

# 前0.8的开盘价和收盘价
for i in range(10, len(train_data)):
    x_train.append(train_data[i - 10:i, 0:2])
    y_train.append(train_data[i, 0])
    if i <= 11:
        print(x_train)
        print(y_train)
        print()

# In[90]:

train_data1 = scaled_data1[0:training_data_len1, :]
x_train1 = []
y_train1 = []

# 前0.8的开盘价和收盘价
for i in range(10, len(train_data1)):
    x_train1.append(train_data1[i - 10:i, 0:2])
    y_train1.append(train_data1[i, 0])
    if i <= 11:
        print(x_train1)
        print(y_train1)
        print()

# In[91]:


train_data2 = scaled_data2[0:training_data_len2, :]
x_train2 = []
y_train2 = []

# 前0.8的开盘价和收盘价
for i in range(10, len(train_data2)):
    x_train2.append(train_data2[i - 10:i, 0:2])
    y_train2.append(train_data2[i, 0])
    if i <= 11:
        print(x_train2)
        print(y_train2)
        print()

# In[92]:


train_data3 = scaled_data3[0:training_data_len3, :]
x_train3 = []
y_train3 = []

# 前0.8的开盘价和收盘价
for i in range(10, len(train_data3)):
    x_train3.append(train_data3[i - 10:i, 0:2])
    y_train3.append(train_data3[i, 0])
    if i <= 11:
        print(x_train3)
        print(y_train3)
        print()

# In[93]:


train_data4 = scaled_data4[0:training_data_len4, :]
x_train4 = []
y_train4 = []

# 前0.8的开盘价和收盘价
for i in range(10, len(train_data4)):
    x_train4.append(train_data4[i - 10:i, 0:2])
    y_train4.append(train_data4[i, 0])
    if i <= 11:
        print(x_train4)
        print(y_train4)
        print()

# In[94]:


x_train, y_train = np.array(x_train), np.array(y_train)
import numpy

# x_train 非目标训练集   y_train   目标训练集
print(x_train)

# In[95]:


x_train1, y_train1 = np.array(x_train1), np.array(y_train1)

# x_train 非目标训练集   y_train   目标训练集
print(x_train1)

# In[96]:


x_train2, y_train2 = np.array(x_train2), np.array(y_train2)

# x_train 非目标训练集   y_train   目标训练集
print(x_train2)

# In[97]:


x_train3, y_train3 = np.array(x_train3), np.array(y_train3)

# x_train 非目标训练集   y_train   目标训练集
print(x_train3)

# In[98]:


x_train4, y_train4 = np.array(x_train4), np.array(y_train4)

# x_train 非目标训练集   y_train   目标训练集
print(x_train4)

# In[99]:


# 创建测试集
test_data = scaled_data[training_data_len - 10:, :]
x_test = []
y_test = []
y_test = scaled_data[training_data_len:, 0:1]
# y_test = dataset[training_data_len:,:]
for i in range(10, len(test_data)):
    x_test.append(test_data[i - 10:i, 0:1])

# In[100]:


# 创建测试集
test_data1 = scaled_data1[training_data_len1 - 10:, :]
x_test1 = []
y_test1 = []
y_test1 = scaled_data1[training_data_len1:, 0:1]
# y_test = dataset[training_data_len:,:]
for i in range(10, len(test_data1)):
    x_test1.append(test_data1[i - 10:i, 0:1])

# In[101]:


# 创建测试集
test_data2 = scaled_data2[training_data_len2 - 10:, :]
x_test2 = []
y_test2 = []
y_test2 = scaled_data2[training_data_len2:, 0:1]
# y_test = dataset[training_data_len:,:]
for i in range(10, len(test_data2)):
    x_test2.append(test_data2[i - 10:i, 0:1])

# In[102]:


# 创建测试集
test_data3 = scaled_data3[training_data_len3 - 10:, :]
x_test3 = []
y_test3 = []
y_test3 = scaled_data3[training_data_len3:, 0:1]
# y_test = dataset[training_data_len:,:]
for i in range(10, len(test_data3)):
    x_test3.append(test_data3[i - 10:i, 0:1])

# In[103]:


# 创建测试集
test_data4 = scaled_data4[training_data_len4 - 10:, :]
x_test4 = []
y_test4 = []
y_test4 = scaled_data4[training_data_len4:, 0:1]
# y_test = dataset[training_data_len:,:]
for i in range(10, len(test_data4)):
    x_test4.append(test_data4[i - 10:i, 0:1])

# In[104]:


# x_test是测试集
x_test = np.array(x_test)
print(x_test)

# In[105]:


x_test1 = np.array(x_test1)
print(x_test1)

# In[106]:


x_test2 = np.array(x_test2)
print(x_test2)

# In[107]:


x_test3 = np.array(x_test3)
print(x_test3)

# In[108]:


x_test4 = np.array(x_test4)
print(x_test4)

# In[109]:

# In[110]:


# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]；输入10个开盘价，预测出第11天的开盘价，循环核时间展开步数为60;
# 每个时间步送入的特征是某一天的开盘价和情感数据，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], 10, 1))
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test = np.reshape(x_test, (x_test.shape[0], 10, 1))

# In[111]:


x_train1 = np.reshape(x_train1, (x_train1.shape[0], 10, 1))
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test1 = np.reshape(x_test1, (x_test1.shape[0], 10, 1))

# In[112]:


x_train2 = np.reshape(x_train2, (x_train2.shape[0], 10, 1))
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test2 = np.reshape(x_test2, (x_test2.shape[0], 10, 1))

# In[113]:


x_train3 = np.reshape(x_train3, (x_train3.shape[0], 10, 1))
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test3 = np.reshape(x_test3, (x_test3.shape[0], 10, 1))

# In[114]:


x_train4 = np.reshape(x_train4, (x_train4.shape[0], 10, 1))
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test4 = np.reshape(x_test4, (x_test4.shape[0], 10, 1))

# In[115]:


model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

# In[123]:


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方差

# In[125]:


model1 = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])
model1.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='mean_squared_error')  # 损失函数用均方差

# In[126]:


model2 = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])
model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='mean_squared_error')  # 损失函数用均方差

# In[127]:


model3 = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])
model3.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='mean_squared_error')  # 损失函数用均方差

# In[128]:


model4 = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])
model4.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='mean_squared_error')  # 损失函数用均方差

# In[129]:


checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

# In[130]:


checkpoint_save_path1 = "./checkpoint/LSTM_stock1.ckpt"

# In[131]:


checkpoint_save_path2 = "./checkpoint/LSTM_stock2.ckpt"

# In[132]:


checkpoint_save_path3 = "./checkpoint/LSTM_stock3.ckpt"

# In[133]:


checkpoint_save_path4 = "./checkpoint/LSTM_stock4.ckpt"

# In[134]:


print(tf.__version__)

# In[135]:


if os.path.exists(checkpoint_save_path + '.index'):
    print('---------------------load the model-----------------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weiths_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

# In[136]:


model.summary()

# In[143]:


if os.path.exists(checkpoint_save_path1 + '.index'):
    print('---------------------load the model-----------------------')
    model.load_weights(checkpoint_save_path1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path1,
                                                 save_weiths_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history1 = model1.fit(x_train1, y_train1, batch_size=16, epochs=10, validation_data=(x_test1, y_test1),
                      validation_freq=1,
                      callbacks=[cp_callback])

# In[144]:


model1.summary()

# In[145]:


if os.path.exists(checkpoint_save_path2 + '.index'):
    print('---------------------load the model-----------------------')
    model.load_weights(checkpoint_save_path2)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path2,
                                                 save_weiths_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history2 = model2.fit(x_train2, y_train2, batch_size=16, epochs=10, validation_data=(x_test2, y_test2),
                      validation_freq=1,
                      callbacks=[cp_callback])

# In[146]:


model2.summary()

# In[147]:


if os.path.exists(checkpoint_save_path3 + '.index'):
    print('---------------------load the model-----------------------')
    model.load_weights(checkpoint_save_path3)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path3,
                                                 save_weiths_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history3 = model3.fit(x_train3, y_train3, batch_size=16, epochs=10, validation_data=(x_test3, y_test3),
                      validation_freq=1,
                      callbacks=[cp_callback])

# In[148]:


model3.summary()

# In[149]:


if os.path.exists(checkpoint_save_path4 + '.index'):
    print('---------------------load the model-----------------------')
    model.load_weights(checkpoint_save_path4)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path4,
                                                 save_weiths_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history4 = model4.fit(x_train4, y_train4, batch_size=16, epochs=10, validation_data=(x_test4, y_test4),
                      validation_freq=1,
                      callbacks=[cp_callback])

# In[150]:


model4.summary()

# In[170]:


file = open('./weight.txt', 'w')

# In[171]:


# In[172]:


# In[173]:


# In[174]:


# In[175]:


for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# In[185]:


file1 = open('./weight1.txt', 'w')
for v in model1.trainable_variables:
    file1.write(str(v.name) + '\n')
    file1.write(str(v.shape) + '\n')
    file1.write(str(v.numpy()) + '\n')
file1.close()

loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']

plt.plot(loss1, label='1Training Loss')
plt.plot(val_loss1, label='1Validation Loss')
plt.title('1Training and Validation Loss')
plt.legend()
plt.show()

# In[189]:


file2 = open('./weight3.txt', 'w')
for v in model2.trainable_variables:
    file2.write(str(v.name) + '\n')
    file2.write(str(v.shape) + '\n')
    file2.write(str(v.numpy()) + '\n')
file2.close()

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

plt.plot(loss2, label='2Training Loss')
plt.plot(val_loss2, label='2Validation Loss')
plt.title('2Training and Validation Loss')
plt.legend()
plt.show()

# In[190]:


file3 = open('./weight3.txt', 'w')
for v in model3.trainable_variables:
    file3.write(str(v.name) + '\n')
    file3.write(str(v.shape) + '\n')
    file3.write(str(v.numpy()) + '\n')
file3.close()

loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']

plt.plot(loss3, label='3Training Loss')
plt.plot(val_loss3, label='3Validation Loss')
plt.title('3Training and Validation Loss')
plt.legend()
plt.show()

# In[191]:


file4 = open('./weight4.txt', 'w')
for v in model4.trainable_variables:
    file4.write(str(v.name) + '\n')
    file4.write(str(v.shape) + '\n')
    file4.write(str(v.numpy()) + '\n')
file4.close()

loss4 = history4.history['loss']
val_loss4 = history4.history['val_loss']

plt.plot(loss4, label='4Training Loss')
plt.plot(val_loss4, label='4Validation Loss')
plt.title('4Training and Validation Loss')
plt.legend()
plt.show()

# In[192]:


predicted_stock_price = model.predict(x_test)

# In[193]:


predicted_stock_price1 = model1.predict(x_test1)

# In[194]:


predicted_stock_price2 = model2.predict(x_test2)

# In[195]:


predicted_stock_price3 = model3.predict(x_test3)

# In[196]:


predicted_stock_price4 = model4.predict(x_test4)

# In[197]:


# 画出真实数据和预测数据的对比曲线

# 准备画布
# fig = plt.figure()
# p = fig.add_subplot(1, 1, 1)

# df['trade_date'] = df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
# print(df)
# plt.figure(figsize=(16, 8))
# plt.title('Close Price History')
# plt.plot(df['trade_date'], df['close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price', fontsize=18)
# plt.show()

def antiNormalize(y_test_t):
    y_test_t = pd.DataFrame(y_test_t)
    y_test_t.columns = ['closed']

    layer = [1 for index in range(len(y_test_t))]
    y_test_t['layer'] = layer
    # y_test_t
    y_test_t = scaler.inverse_transform(y_test_t)
    # y_test_t
    ytest = y_test_t
    ytest = pd.DataFrame(ytest)
    ytest.columns = ['close', 'scores']
    ytest = ytest.drop(['scores'], axis=1)
    ytest = np.array(ytest)
    return ytest


def antiNormalizePredict(predicted_stock_price):
    predicted_stock_pricepd = pd.DataFrame(predicted_stock_price)
    predicted_stock_pricepd.columns = ['closed']

    layer = [1 for index in range(len(predicted_stock_pricepd))]
    predicted_stock_pricepd['layer'] = layer
    # predicted_stock_pricepd
    predicted_stock_pricepd = np.array(predicted_stock_pricepd)
    predicted_stock_pricepd = scaler.inverse_transform(predicted_stock_pricepd)
    predicted_stock_pricepd111 = pd.DataFrame(predicted_stock_pricepd)
    predict = predicted_stock_pricepd111
    predict.columns = ['close', 'scores']
    predict = predict.drop(['scores'], axis=1)
    predict = np.array(predict)
    return predict

y_test = antiNormalize(y_test)
y_test1 = antiNormalize(y_test1)
y_test2 = antiNormalize(y_test2)
y_test3 = antiNormalize(y_test3)
y_test4 = antiNormalize(y_test4)

predicted_stock_price = antiNormalizePredict(predicted_stock_price)
predicted_stock_price1 = antiNormalizePredict(predicted_stock_price1)
predicted_stock_price2 = antiNormalizePredict(predicted_stock_price2)
predicted_stock_price3 = antiNormalizePredict(predicted_stock_price3)
predicted_stock_price4 = antiNormalizePredict(predicted_stock_price4)

time_data = pd.DataFrame(time_data)


y_test0 = pd.DataFrame(y_test)
predicted_stock_price0 = pd.DataFrame(predicted_stock_price)
time_data.columns=['time_data']
y_test0.columns=['y_test0']
predicted_stock_price0.columns=['predicted_stock_price0']
Sinopec = pd.concat([time_data, predicted_stock_price0, y_test0],axis=1)
plt.figure(figsize=(16,8))
plt.title('Sinopec Stock Price')
plt.plot(Sinopec['time_data'], Sinopec['predicted_stock_price0'])
plt.plot(Sinopec['time_data'], Sinopec['y_test0'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()


##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, y_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, y_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, y_test)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)

# In[202]:


# 画出真实数据和预测数据的对比曲线
y_test11 = pd.DataFrame(y_test1)
predicted_stock_price11 = pd.DataFrame(predicted_stock_price1)
time_data.columns=['time_data']
y_test11.columns=['y_test11']
predicted_stock_price11.columns=['predicted_stock_price11']
Sinopec = pd.concat([time_data, predicted_stock_price11, y_test11],axis=1)
plt.figure(figsize=(16,8))
plt.title('PetroChina Stock Price')
plt.plot(Sinopec['time_data'], Sinopec['predicted_stock_price11'])
plt.plot(Sinopec['time_data'], Sinopec['y_test11'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse1 = mean_squared_error(predicted_stock_price1, y_test1)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse1 = math.sqrt(mean_squared_error(predicted_stock_price1, y_test1))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae1 = mean_absolute_error(predicted_stock_price1, y_test1)
print('均方误差: %.6f' % mse1)
print('均方根误差: %.6f' % rmse1)
print('平均绝对误差: %.6f' % mae1)

# In[203]:


# 画出真实数据和预测数据的对比曲线
y_test22 = pd.DataFrame(y_test2)
predicted_stock_price22 = pd.DataFrame(predicted_stock_price2)
time_data.columns=['time_data']
y_test22.columns=['y_test22']
predicted_stock_price22.columns=['predicted_stock_price22']
Sinopec = pd.concat([time_data, predicted_stock_price22, y_test22],axis=1)
plt.figure(figsize=(16,8))
plt.title('COSL Stock Price')
plt.plot(Sinopec['time_data'], Sinopec['predicted_stock_price22'])
plt.plot(Sinopec['time_data'], Sinopec['y_test22'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse2 = mean_squared_error(predicted_stock_price2, y_test2)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse2 = math.sqrt(mean_squared_error(predicted_stock_price2, y_test2))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae2 = mean_absolute_error(predicted_stock_price2, y_test2)
print('均方误差: %.6f' % mse2)
print('均方根误差: %.6f' % rmse2)
print('平均绝对误差: %.6f' % mae2)

# In[204]:


# 画出真实数据和预测数据的对比曲线
y_test33 = pd.DataFrame(y_test3)
predicted_stock_price33 = pd.DataFrame(predicted_stock_price3)
time_data.columns=['time_data']
y_test33.columns=['y_test33']
predicted_stock_price33.columns=['predicted_stock_price33']
Sinopec = pd.concat([time_data, predicted_stock_price33, y_test33],axis=1)
plt.figure(figsize=(16,8))
plt.title('SStsP Stock Price')
plt.plot(Sinopec['time_data'], Sinopec['predicted_stock_price33'])
plt.plot(Sinopec['time_data'], Sinopec['y_test33'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse3 = mean_squared_error(predicted_stock_price3, y_test3)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse3 = math.sqrt(mean_squared_error(predicted_stock_price3, y_test3))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae3 = mean_absolute_error(predicted_stock_price3, y_test3)
print('均方误差: %.6f' % mse3)
print('均方根误差: %.6f' % rmse3)
print('平均绝对误差: %.6f' % mae3)

# In[205]:


# 画出真实数据和预测数据的对比曲线
y_test44 = pd.DataFrame(y_test4)
predicted_stock_price44 = pd.DataFrame(predicted_stock_price4)
time_data.columns=['time_data']
y_test44.columns=['y_test44']
predicted_stock_price44.columns=['predicted_stock_price44']
Sinopec = pd.concat([time_data, predicted_stock_price44, y_test44],axis=1)
plt.figure(figsize=(16,8))
plt.title('MMPetro Stock Price')
plt.plot(Sinopec['time_data'], Sinopec['predicted_stock_price44'])
plt.plot(Sinopec['time_data'], Sinopec['y_test44'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse4 = mean_squared_error(predicted_stock_price4, y_test4)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse4 = math.sqrt(mean_squared_error(predicted_stock_price4, y_test4))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae4 = mean_absolute_error(predicted_stock_price4, y_test4)
print('均方误差: %.6f' % mse4)
print('均方根误差: %.6f' % rmse4)
print('平均绝对误差: %.6f' % mae4)

# In[209]:


##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, y_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, y_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, y_test)

mse1 = mean_squared_error(predicted_stock_price1, y_test1)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse1 = math.sqrt(mean_squared_error(predicted_stock_price1, y_test1))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae1 = mean_absolute_error(predicted_stock_price1, y_test1)

mse2 = mean_squared_error(predicted_stock_price2, y_test2)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse2 = math.sqrt(mean_squared_error(predicted_stock_price2, y_test2))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae2 = mean_absolute_error(predicted_stock_price2, y_test2)

mse3 = mean_squared_error(predicted_stock_price3, y_test3)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse3 = math.sqrt(mean_squared_error(predicted_stock_price3, y_test3))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae3 = mean_absolute_error(predicted_stock_price3, y_test3)

mse4 = mean_squared_error(predicted_stock_price4, y_test4)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse4 = math.sqrt(mean_squared_error(predicted_stock_price4, y_test4))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae4 = mean_absolute_error(predicted_stock_price4, y_test4)

print('中国石化均方误差: %.6f    中国石化均方根误差: %.6f      中国石化平均绝对误差: %.6f' % (mse, rmse, mae))
print('中国石油均方误差: %.6f    中国石油均方根误差: %.6f      中国石油平均绝对误差: %.6f' % (mse1, rmse1, mae1))
print('中国油服均方误差: %.6f    中国油服均方根误差: %.6f      中国油服平均绝对误差: %.6f' % (mse2, rmse2, mae2))
print('泰山石油均方误差: %.6f    泰山石油均方根误差: %.6f      泰山石油平均绝对误差: %.6f' % (mse3, rmse3, mae3))
print('茂化实华均方误差: %.6f    茂化实华均方根误差: %.6f      茂化实华平均绝对误差: %.6f' % (mse4, rmse4, mae4))

