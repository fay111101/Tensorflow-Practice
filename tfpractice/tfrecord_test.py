#!/usr/bin/env python
# encoding: utf-8
"""
@author: fay
@contact: fayfeixiuhong@didiglobal.com
@software: pycharm
@file: tfrecord_test.py
@time: 2019-11-04 17:25
@desc:
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# mnist = read_data_sets("MNIST_data/", one_hot=True)


# 把数据写入Example
def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    feat_shape = feature.shape
    tfrecords_features['feature'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    tfrecords_features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label))
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


# 把所有数据写入tfrecord文件
def make_tfrecord(data, outf_nm='mnist-train'):
    feats, labels = data
    outf_nm += '.tfrecord'
    tfrecord_wrt = tf.python_io.TFRecordWriter(outf_nm)
    ndatas = len(labels)
    for inx in range(ndatas):
        exmp = get_tfrecords_example(feats[inx], labels[inx])
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()


# import random

# nDatas = len(mnist.train.labels)
# inx_lst = list(range(nDatas))
# random.shuffle(inx_lst)
# random.shuffle(inx_lst)
# ntrains = int(0.85 * nDatas)
#
# # make training set
# data = ([mnist.train.images[i] for i in inx_lst[:ntrains]], \
#         [mnist.train.labels[i] for i in inx_lst[:ntrains]])
# make_tfrecord(data, outf_nm='./MNIST_tfrecord/mnist-train')
#
# # make validation set
# data = ([mnist.train.images[i] for i in inx_lst[ntrains:]], \
#         [mnist.train.labels[i] for i in inx_lst[ntrains:]])
# make_tfrecord(data, outf_nm='./MNIST_tfrecord/mnist-val')
#
# # make test set
# data = (mnist.test.images, mnist.test.labels)
# make_tfrecord(data, outf_nm='./MNIST_tfrecord/mnist-test')
#
train_f, val_f, test_f = ['./MNIST_tfrecord/mnist-%s.tfrecord' % i for i in ['train', 'val', 'test']]


def parse_example(serial_exmp):
    feats = tf.parse_single_example(serial_exmp, features={'feature': tf.FixedLenFeature([], tf.string), \
                                                           'label': tf.FixedLenFeature([10], tf.float32),
                                                           'shape': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(feats['feature'], tf.float32)
    label = feats['label']
    shape = tf.cast(feats['shape'], tf.int32)
    return image, label, shape


def get_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(parse_example)


epochs = 16
batch_size = 50

# training dataset
nDatasTrain = 46750
dataset_train = get_dataset(train_f)
# make sure repeat is ahead batch
# this is different from dataset.shuffle(1000).batch(batch_size).repeat(epochs)
# the latter means that there will be a batch data with nums less than batch_size for each epoch
# if when batch_size can't be divided by nDatas.
dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
nBatchs = nDatasTrain * epochs // batch_size
print ('nBatchs',nBatchs)

# evalation dataset
nDatasVal = 8250
dataset_val = get_dataset(val_f)
dataset_val = dataset_val.batch(nDatasVal).repeat(nBatchs // 100 * 2)

# test dataset
nDatasTest = 10000
dataset_test = get_dataset(test_f)
dataset_test = dataset_test.batch(nDatasTest)

# make dataset iterator
iter_train = dataset_train.make_one_shot_iterator()
iter_val = dataset_val.make_one_shot_iterator()
iter_test = dataset_test.make_one_shot_iterator()

# make feedable iterator, i.e. iterator placeholder
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, \
                                               dataset_train.output_types, dataset_train.output_shapes)
x, y_, _ = iterator.get_next()

# cnn
x_image = tf.reshape(x, [-1, 28, 28, 1])
w_init = tf.truncated_normal_initializer(stddev=0.1, seed=9)
b_init = tf.constant_initializer(0.1)
cnn1 = tf.layers.conv2d(x_image, 32, (5, 5), padding='same', activation=tf.nn.relu, \
                        kernel_initializer=w_init, bias_initializer=b_init)
mxpl1 = tf.layers.max_pooling2d(cnn1, 2, strides=2, padding='same')
cnn2 = tf.layers.conv2d(mxpl1, 64, (5, 5), padding='same', activation=tf.nn.relu, \
                        kernel_initializer=w_init, bias_initializer=b_init)
mxpl2 = tf.layers.max_pooling2d(cnn2, 2, strides=2, padding='same')
mxpl2_flat = tf.reshape(mxpl2, [-1, 7 * 7 * 64])
fc1 = tf.layers.dense(mxpl2_flat, 1024, activation=tf.nn.relu, \
                      kernel_initializer=w_init, bias_initializer=b_init)
keep_prob = tf.placeholder('float')
fc1_drop = tf.nn.dropout(fc1, keep_prob)
logits = tf.layers.dense(fc1_drop, 10, kernel_initializer=w_init, bias_initializer=b_init)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
optmz = tf.train.AdamOptimizer(1e-4)
train_op = optmz.minimize(loss)


def get_eval_op(logits, labels):
    corr_prd = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(corr_prd, 'float'))


eval_op = get_eval_op(logits, y_)

init = tf.global_variables_initializer()

# summary
logdir = './logs/m4d2a'


def summary_op(datapart='train'):
    tf.summary.scalar(datapart + '-loss', loss)
    tf.summary.scalar(datapart + '-eval', eval_op)
    return tf.summary.merge_all()


summary_op_train = summary_op()
summary_op_val = summary_op('val')

# whether to restore or not
ckpts_dir = 'ckpts/'
ckpt_nm = 'cnn-ckpt'
saver = tf.train.Saver(max_to_keep=50)  # defaults to save all variables, using dict {'x':x,...} to save specified ones.
restore_step = 'latest'
start_step = 0
train_steps = nBatchs
best_loss = 1e6
best_step = 0

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth=True # allocate when needed
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(init)
    handle_train, handle_val, handle_test = sess.run( \
        [x.string_handle() for x in [iter_train, iter_val, iter_test]])
    if restore_step:
        ckpt = tf.train.get_checkpoint_state(ckpts_dir)
        if ckpt and ckpt.model_checkpoint_path:  # ckpt.model_checkpoint_path means the latest ckpt
            if restore_step == 'latest':
                ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
                start_step = int(ckpt_f.split('-')[-1]) + 1
                # print ('ckpt-split',ckpt_f.split('-'))
            else:
                ckpt_f = ckpts_dir + ckpt_nm + '-' + restore_step
            print('loading wgt file: ' + ckpt_f)
            saver.restore(sess, ckpt_f)
    summary_wrt = tf.summary.FileWriter(logdir, sess.graph)
    if restore_step in ['', 'latest']:
        for i in range(start_step, train_steps):
            _, cur_loss, cur_train_eval, summary = sess.run([train_op, loss, eval_op, summary_op_train], \
                                                            feed_dict={handle: handle_train, keep_prob: 0.5})
            # log to stdout and eval validation set
            if i % 100 == 0 or i == train_steps - 1:
                saver.save(sess, ckpts_dir + ckpt_nm, global_step=i)  # save variables
                summary_wrt.add_summary(summary, global_step=i)
                cur_val_loss, cur_val_eval, summary = sess.run([loss, eval_op, summary_op_val], \
                                                               feed_dict={handle: handle_val, keep_prob: 1.0})
                if cur_val_loss < best_loss:
                    best_loss = cur_val_loss
                    best_step = i
                summary_wrt.add_summary(summary, global_step=i)
                print('step %5d: loss %.5f, acc %.5f --- loss val %0.5f, acc val %.5f' % (i, \
                                                                                          cur_loss, cur_train_eval,
                                                                                  cur_val_loss,
                                                                                  cur_val_eval))
                # sess.run(init_train)
        with open(ckpts_dir + 'best.step', 'w') as f:
            f.write('best step is %d\n' % best_step)
        print('best step is %d' % best_step)
    # eval test set
    test_loss, test_eval = sess.run([loss, eval_op], feed_dict={handle: handle_test, keep_prob: 1.0})
    print('eval test: loss %.5f, acc %.5f' % (test_loss, test_eval))
