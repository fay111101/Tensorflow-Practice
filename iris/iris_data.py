#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-6-28 下午2:34
@Author  : fay
@Email   : fay625@sina.cn
@File    : iris_data.py
@Software: PyCharm
"""
import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, test_path


def load_data(label_name='Species'):
    """
    :param y_name: label
    :return: (train_x,train_y)(test_x,test_y)
    """
    # # train_path,test_path=download()
    # train_path='./data/iris_training.csv'
    # test_path='./data/iris_test.csv'
    # train=pd.read_csv(train_path,names=CSV_COLUMN_NAMES)
    # train_x,train_y=train,train.pop(y_name)
    #
    # test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    # test_x, test_y = test, test.pop(y_name)
    #
    # return (train_x,train_y),(test_x,test_y)
    # Create a local copy of the training set.


    # train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
    #                                      origin=TRAIN_URL)
    train_path='./data/iris_training.csv'
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                        )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)
    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)


def train_input_fn(features, labels, batch_size):
    """
    :param features:
    :param labels:
    :param batch_size:
    :return:
    """
    # from_tensor_slices函数接受一个数组并返回表示该数组切片的tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle,repeat,and batch the examples
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # 创建迭代器，并返回
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
