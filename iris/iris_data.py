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
TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
TEST_URL="http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES=['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']


def download():
    train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
    test_path=tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

    return train_path,test_path

def load_data(y_name='Species'):
    """

    :param y_name: label
    :return: (train_x,train_y)(test_x,test_y)
    """
    train_path,test_path=download()
    train=pd.read_csv(test_path,names=CSV_COLUMN_NAMES)
    train_x,train_y=train,train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x,train_y),(test_x,test_y)

def train_input_fn(features,labels,batch_size):
    """

    :param features:
    :param labels:
    :param batch_size:
    :return:
    """
    # from_tensor_slices函数接受一个数组并返回表示该数组切片的tf.data.Dataset
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    # Shuffle,repeat,and batch the examples
    dataset=dataset.shuffle(1000).repeat().batch(batch_size)
    # 创建迭代器，并返回
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
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
    fields=tf.decode_csv(line,record_defaults=CSV_TYPES)
    features=dict(zip(CSV_COLUMN_NAMES,fields))
    label=features.pop('Species')
    return features,label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

