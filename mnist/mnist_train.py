#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-5 下午10:39
@Author  : fay
@Email   : fay625@sina.cn
@File    : mnist_train.py
@Software: PyCharm
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist_inference
#
# BATCH_SIZE = 100
# # 模型相关的参数
# LEARNING_RATE_BASE = 0.8  # 基础学习率
# LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
# REGULARAZTION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
# # TRAINING_STEPS = 5000  # 训练轮数
# TRAINING_STEPS = 30000  # 训练轮数
# MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
#
# # 模型保存的路径和文件名
# MODEL_SAVE_PATH = './model/'
# MODEL_NAME = 'model.ckpt'
#
#
# def train(mnist):
#     x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
#     y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')
#
#     regualizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
#     y = mnist_inference.inference1(x, regualizer)
#     global_step = tf.Variable(0, trainable=False)
#     variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#     variable_average_op = variable_average.apply(tf.trainable_variables())
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#     learning_rate = tf.train.exponential_decay(
#         LEARNING_RATE_BASE,
#         global_step,
#         mnist.train.num_examples / BATCH_SIZE,
#         LEARNING_RATE_DECAY,
#         staircase=True)
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#     with tf.control_dependencies([train_step, variable_average_op]):
#         train_op = tf.no_op(name='train')
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#
#         for i in range(TRAINING_STEPS):
#             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={
#                 x: xs, y_: ys
#             })
#
#             if i % 1000 == 0:
#                 print("After %d trainning steps,loss on trainning batch is %g" % (step, loss_value))
#                 saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
#
#
# def main(argv=None):
#     # mnist = input_data.read_data_sets('MINIST_data', one_hot=True)
#     mnist = read_data_sets('MINIST_data', one_hot=True)
#     train(mnist)
#
#
# if __name__ == '__main__':
#     main()
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train(mnist):
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference1(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("MINIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
