#!/usr/bin/env python
# encoding: utf-8
"""
@author: fay
@contact: fayfeixiuhong@didiglobal.com
@software: pycharm
@file: l2.py
@time: 2019-04-12 16:37
@desc:
"""

import tensorflow as tf


def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 把正则化加入集合losses里面
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 真值

batcg_size = 8

layer_dimension = [2, 10, 10, 10, 1]  # 神经网络层节点的个数

n_layers = len(layer_dimension)  # 神经网络的层数

cur_layer = x

in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, tf.shape(out_dimension)))
    cur_layer = tf.nn.relu(tf.matmul(x, weight) + bias)

in_dimension = layer_dimension[i]

ses_loss = tf.reduce_mean(tf.square(y_ - cur_layer))  # 计算最终输出与标准之间的loss

tf.add_to_collenction("losses", ses_loss)  # 把均方误差也加入到集合里

loss = tf.add_n(tf.get_collection("losses"))
# tf.get_collection返回一个列表,内容是这个集合的所有元素
# add_n()把输入按照元素相加
