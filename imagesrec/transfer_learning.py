#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-6 下午2:14
@Author  : fay
@Email   : fay625@sina.cn
@File    : transfer_learning.py
@Software: PyCharm
"""

'''
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
curl http://download.tensorflow.org/example_images/flower_photos.tgz
'''
import glob
import os.path
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = './data/inception_dec_2015'
MODEL_FILE = 'tensorflow_inception_graph.pb'

CACHE_DIR = './data/bottleneck'
INPUT_DATA = './data/flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


def create_image_lists(test_percentage, validation_percentage):
    '''
    把样本中所有的图片列表并按训练、验证、测试数据分开
    :param test_percentage:
    :param validation_percentage:
    :return:
    '''
    result = {}
    # 获取当前目录下所有的子目录，其中包括当前目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    # 剔除当前目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取目录下所有有效的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        # 通过目录名获取类别名
        label_name = dir_name.lower()

        train_images = []
        test_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (test_percentage + validation_percentage):
                test_images.append(base_name)
            else:
                train_images.append(base_name)
        # 将当前类别的数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': train_images,
            'testing': test_images,
            'validation': validation_images
        }
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    '''
    通过类别名称、所属数据集和图片编号获取一张图片的地址。
    :param image_lists: 所有图片信息
    :param image_dir: 存放图片的根目录
    :param label_name: 类别名称
    :param index: 需要获取的图片的编号
    :param category: 指定需要获取的图片是训练集，测试集还是验证集中的
    :return:
    '''
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 为数据根目录的地址+类别的文件夹+图片名
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    '''
    定义函数获取Inception-v3模型处理之后的特征向量的文件地址。
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :return:
    '''
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    '''
    使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过神经网络处理的结果是一个四维数组,压缩成一维特征向量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    '''
    先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存在文件中
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    '''
    这个函数随机获取一个batch的图片作为训练数据。
    :param sess:
    :param n_classes:
    :param image_lists:
    :param how_many:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入到当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        # 是该类的位置为1，其他位置为0
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    '''
    这个函数获取全部的测试数据，并计算正确率
    :param sess:
    :param image_lists:
    :param n_classes:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))

        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
