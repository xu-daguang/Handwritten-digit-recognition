
#载入套件
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#载入资料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#for i in range(5*5):
#    plt.subplot(5, 5, i+1)
#    plt.imshow(x_train[i], cmap='gray')
#    plt.title('lable = {}'.format(y_train[i]))
#    plt.axis('off')
#    plt.tight_layout()
#    plt.show()

#print(x_train[3])

#print(x_train.shape[0],'张训练图')
#print(x_test.shape[0],'张测试图')

#print('训练资料影像大小=',x_train.shape)
#print('测试资料标记值的个数=', len(y_train.shape))

#形态处理
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


#训练
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28*28,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=2, validation_data=(x_test,y_test))

model.save('手写数字辨识模型.h5')


#预测
score = model.evaluate(x_test, y_test, verbose=2)
print('测试结果和标准答案的误差：', score[0])
print('测试的辨识率：', score[1])


