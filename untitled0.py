# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-IX1hY9-AwprT0YUvUdsPMNwn3jqjQao
"""

1+1  #许大光

1/2

print('Hello World!')

import numpy as np
import pandas as pd  #要將label轉成one-hot encoding
from keras.utils import np_utils    #
np.random.seed(10)   #為了教學方便

from keras.datasets import mnist    #準備匯入資料集mnist

(X_train_image,Y_train_lable),(X_test_image,Y_test_lable)=mnist.load_data()

X_train_image.shape
X_train_image[0:2]
print('lable;',Y_train_lable.shape)
Y_train_lable[0:5]

import matplotlib.pyplot as plt
def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(2,2)
  plt.imshow(image,cmap = 'binary')
  plt.show()

for x in range(0,5):
  plot_image(X_train_image[x])

#进行转换成一维
#將數據類型轉為浮點型 範圍為0-1之間
X_train = X_train_image.reshape(60000,28*28).astype('float32')
X_test = X_test_image.reshape(10000,28*28).astype('float32')
X_train_normalization = X_train / 255    #正规化0-1之间
X_test_normalization = X_test / 255

#ine hot edcoding:2-> 0010000000
Y_train_onehot = np_utils.to_categorical(Y_train_lable)
Y_test_onehot = np_utils.to_categorical(Y_test_lable)
# 0.2 0.4 0.1 0.2....

#MLP
#model1建立
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units = 256,
               input_dim = 784,
               kernel_initializer = 'normal',
               activation = 'relu'))

model.add(Dense(units = 10,
               kernel_initializer = 'normal',
               activation = 'softmax'))

model.summary()

#定义训练方式
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#开始训练
train_history = model.fit(x = X_train_normalization,
                         y = Y_train_onehot,
                         validation_split = 0.2, 
                         epochs = 10,
                         batch_size = 200,
                         verbose = 2)

Y_testOneHOT = np_utils.to_categorical(Y_test_lable)
score = model.evaluate(X_test_normalization,Y_testOneHOT)
print('正確度 :',score[1]*100,'%')

#混淆矩阵
prediction = model.predict_classes(X_test)
pd.crosstab(Y_test_lable,prediction,
           rownames = ['lable'],colnames = ['predict'])

#
df = pd.DataFrame({'lable' :Y_test_lable,'predict' :prediction})
print(df)

df_8to3 = df[(df.lable == 5)&(df.predict == 3)]

plot_image(X_test_image[6755])

for x in df_8to3.index:
  plot_image(X_test_image[x])