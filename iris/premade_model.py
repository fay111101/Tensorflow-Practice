#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-6-28 下午1:15
@Author  : fay
@Email   : fay625@sina.cn
@File    : premade_model.py
@Software: PyCharm
@Description: An Example of a DNNClassifier for the Iris dataset
"""
import argparse
import tensorflow as tf
import iris.iris_data as iris_data
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',default=100,type=int,help='batch_size')
parser.add_argument('--train_size',default=100,type=int,help='number of trainning steps')

def main(argv):
    args=parser.parse_args(argv[1:])
    (train_x,train_y),(test_x,test_y)=iris_data.load_data()
    my_feature_columns=[]
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))



