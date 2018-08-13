#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-10 下午4:27
@Author  : fay
@Email   : fay625@sina.cn
@File    : keras_cnn.py
@Software: PyCharm
"""
import keras

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

num_classes = 10
img_rows, img_cols = 28, 28
# 1.数据处理
# 通过Keras封装好的API加载MNIST数据。其中trainX就是一个60000 * 28 * 28的数组，
# trainY是每一张图片对应的数字。

(trainX, trainY), (testX, testY) = mnist.load_data()

# def load_mnist(path, kind='train'):
#     """Load MNIST data from `path`"""
#     labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
#     images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
#     with open(labels_path, 'rb') as lbpath:
#         magic, n = struct.unpack('>II', lbpath.read(8))
#         labels = np.fromfile(lbpath, dtype=np.uint8)
#     with open(images_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
#         images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
#     return images, labels


# X_train, y_train = load_mnist('../data', kind='train')
# print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
# X_test, y_test = load_mnist('../data', kind='t10k')
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# 根据对图像编码的格式要求来设置输入层的格式。
if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

# 将标准答案转化为需要的格式（one-hot编码）。
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

# 2.模型定义
# 使用Keras API定义模型。
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 定义损失函数、优化函数和评测方法。
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
# 3.通过Keras的API训练模型并计算在测试数据上的准确率。
model.fit(trainX, trainY, batch_size=128, epochs=10, validation_data=(testX, testY))
# 在测试数据上计算准确率。
score = model.evaluate(testX, testY)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
