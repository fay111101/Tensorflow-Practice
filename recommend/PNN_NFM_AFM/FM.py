#!/usr/bin/env python
# encoding: utf-8
"""
@author: fay
@contact: fayfeixiuhong@didiglobal.com
@software: pycharm
@file: FM.py
@time: 2019-11-06 14:35
@desc:
"""
import functools
import tensorflow as tf
import numpy as np
import os
import pandas as pd


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class FM_FTRL:
    def __init__(self, x, y, p, k):
        """

        :param x: input x
        :param y: label
        :param p: number of columns
        :param k: dim of v for FM pair interaction vector
        """
        self.x = x
        self.y = y
        self.p = p
        self.k = k
        self.predict
        self.optimize
        self.w0
        self.W
        self.V
        self.norm
        self.error
        self.loss

    @define_scope
    def predict(self):
        """
        this function used to predict data
        :return:
        """
        x = self.x
        self.w0 = tf.Variable(tf.zeros([1]))
        self.W = tf.Variable(tf.zeros([self.p]))
        self.V = tf.Variable(tf.random_normal([self.k, self.p], stddev=0.01))
        liner_terms = tf.add(self.w0,
                             tf.reduce_sum(tf.multiply(self.W, x), 1, keepdims=True)
                             )
        pair_terms = tf.multiply(0.5,
                                 tf.reduce_sum(
                                     tf.subtract(
                                         tf.pow(tf.matmul(x, tf.transpose(self.V)), 2),
                                         tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(self.V, 2)))
                                     )
                                 ))
        predict = tf.add(liner_terms, pair_terms)
        return predict

    @define_scope
    def norm(self):
        """

        :return:
        """
        lambda_w = tf.constant(0.001, name="lambda_w")
        lambda_v = tf.constant(0.001, name="lambda_v")
        l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w, tf.pow(self.W, 2)),
                tf.multiply(lambda_v, tf.pow(self.V, 2))
            )
        )
        return l2_norm

    @define_scope
    def error(self):
        y = self.y
        y_hat = self.predict
        error = tf.reduce_mean(
            tf.square(
                tf.subtract(y, y_hat)
            )
        )
        return error

    @define_scope
    def loss(self):
        loss = tf.add(self.error, self.norm)
        return loss

    @define_scope
    def optimize(self, mode="ftrl"):
        if mode == 'ftrl':
            opt = tf.train.FtrlOptimizer(learning_rate=0.1).minimize(self.loss)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        return opt


def hash_java(key):
    """
    hash equal to jaha hash funtion,which hash valus > 0, this is very import for engineer and ont-hot encode
    :param key:
    :return:
    """
    h = 0
    for c in key:
        h = ((h * 37) + ord(c)) & 0xFFFFFFFF
    return h


def main():
    """

    :return:
    """
    epochs = 20
    batch_size = 1000

    D = 3000
    p = 2
    k = 2

    cols = ['user', 'item', 'rating', 'timestamp']
    use_cols = ['user', 'item', 'rating']
    features = ['user', 'item']

    # /Users/didi/Project/Tensorflow-Practice/recommend
    data_dir = os.path.abspath(f"{os.path.abspath(os.path.dirname(os.path.realpath(__file__)))}/../../Tensorflow-Practice/recommend/data/ml-100k")
    x = tf.placeholder('float', shape=[None, D])
    y = tf.placeholder('float', shape=[None, 1])
    model = FM_FTRL(x=x, y=y, p=D, k=k)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num_lines = sum(1 for l in open(f'{data_dir}/ua.base')) - 1
    print(f"total train lines number is {num_lines}")
    for epoch in range(epochs):
        total_bacth = 0
        avg_cost = 0.
        # create random data based on random index
        index_random = np.random.permutation(num_lines)

        for row_index in range(0, index_random.shape[0], batch_size):
            skip_rows = np.concatenate([index_random[:row_index], index_random[row_index+batch_size:]])
            row = pd.read_csv(f'{data_dir}/ua.base', delimiter='\t', names=cols,
                              usecols=['user', 'item', 'rating'],
                              skiprows=skip_rows)
            total_bacth += 1
            bY = row['rating'].values.reshape(-1, 1)
            bX = np.zeros([D])
            for f in features:
                hash_index = hash_java(str(row[f]) + f) % D
                if hash_index < 0:
                    raise ValueError("index for one-hot should be bigger than 0")
                bX[hash_index] = 1
            bX = bX.reshape(-1, D)
            mse, loss_val, w, v, _ = sess.run([model.error, model.loss, model.W, model.V, model.optimize],
                                              feed_dict={x: bX, y: bY})
            avg_cost += loss_val
            # Display logs per epoch step
        if (epoch + 1) % 1 == 0:
            print(f"total batch is {total_bacth}")
            print(f"Epoch:{epoch + 1}, cost={avg_cost/total_bacth}")
    print('MSE: ', mse)
    print('Learnt weights:', w, w.shape)
    print('Learnt factors:', v, v.shape)
    # print(f"auc value is {tf.summary.scalar('AUC', auc)}")

    errors = []
    test = pd.read_csv(f'{data_dir}/ua.test', delimiter='\t', names=cols, usecols=['user', 'item', 'rating'],
                       chunksize=batch_size)
    for row in test:
        bY = row['rating'].values.reshape(-1, 1)
        bX = np.zeros([D])
        for f in features:
            hash_index = hash_java(str(row[f]) + "_" + f) % D
            bX[hash_index] = 1
        bX = bX.reshape(-1, D)
        errors.append(sess.run(model.error, feed_dict={x: bX, y: bY}))

    RMSE = np.sqrt(np.array(errors).mean())
    print(RMSE)
    sess.close()


if __name__ == '__main__':
    main()