#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-9 下午3:05
@Author  : fay
@Email   : fay625@sina.cn
@File    : predict_sin.py TF1.8
@Software: PyCharm
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import rnn

'''
参考
https://blog.csdn.net/yunge812/article/details/79444089
'''
HIDDEN_SIZE = 30  # LSTM中隐藏节点的数目
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 10  # 训练序列长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小
TRAIN_EXAMPLES = 10000  # 训练数据个数
TEST_EXAMPLES = 1000
SAMPLE_GAP = 0.01  # 采样间隔


def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    return lstm_cell


# 定义lstm模型
def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)
    # train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
    #                                            optimizer="Adagrad",
    #                                            learning_rate=0.1)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer="Adagrad",
        learning_rate=0.1)
    return predictions, loss, train_op


def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果，
    with tf.variable_scope('model', reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    predictions = []
    labels = []
    for i in range(TEST_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    print(predictions)
    print(labels)
    # 计算rmse
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt((predictions - labels) ** 2).mean(axis=0)
    print("Root Mean Square Error is :%f", rmse)

    plt.figure()
    plt.plot(predictions, labels='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices(
        {
            'train_X': train_X,
            'train_y': train_y
        })
    ds = ds.repeat().shuffle(buffer_size=1000).batch(batch_size=BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    # 定义模型，得到预测结果，损失函数和训练操作
    with tf.variable_scope('model'):
        predictions, loss, train_op = lstm_model(X, y)

    sess.run(tf.global_variables_initializer())
    # 训练模型
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print('train step: ' + str(i) + ",loss:" + str(l))



test_start = TRAIN_EXAMPLES * SAMPLE_GAP
test_end = (TRAIN_EXAMPLES + TEST_EXAMPLES) * SAMPLE_GAP
# 产生仿真数据x  y 用于训练
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start,
                                                    TRAIN_EXAMPLES, dtype=np.float32)))
# 产生仿真数据x  y 用于测试
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end,
                                                  TEST_EXAMPLES, dtype=np.float32)))

learn = tf.contrib.learn
regressor = learn.Estimator(model_fn=lstm_model)
regressor.fit(train_x, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
predicted = [[pred] for pred in regressor.predict(test_x)]
rmse = np.sqrt((predicted - test_y) ** 2).mean(axis=0)
fig = plt.figure()
plot_predicted = plt.plot(predicted, label="predicted", color='red')
plot_test = plt.plot(test_y, label="real_sin", color='blue')
plt.legend = ([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()


# with tf.Session() as sess:
#     train(sess, train_x, train_y)
#     run_eval(test_x, test_y)
'''
ValueError: Expected input tensor Tensor("model/rnn/Const:0", shape=(), dtype=string) to have rank at least 2
The error seems fairly clear, tf.nn.dynamic_rnn expects a 3-dimensional tensor as input (i.e. rank 3), 
but fc2 has only two dimensions. 
The shape of fc2 should be something like (<batch_size>, <max_time>, <num_features>) (or (<max_time>, <batch_size>, 
<num_features>) if you pass time_major=True)
'''
