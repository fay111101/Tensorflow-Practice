#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-13 下午3:26
@Author  : fay
@Email   : fay625@sina.cn
@File    : AE.py
@Software: PyCharm
"""
'''
参考深度学习书籍自编码器
https://blog.csdn.net/Vinsuan1993/article/details/81142855
InternalError (see above for traceback): Blas GEMM launch failed : a.shape=(256, 784), b.shape=(784, 256), m=256, n=256, k=784
	 [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](_arg_Placeholder_0_0/_1, Variable/read)]]
'''
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets('data',one_hot=True)
learning_rate=0.01
n_hidden_1=256
n_hidden_2=128
n_input=784

x=tf.placeholder(dtype=np.float32,shape=[None,n_input])
y=x

weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}

bias={
    'encoder_b1':tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.zeros([n_input])),
}

def encoder(x):
    layer1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),bias['encoder_b1']))
    layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_h2']),bias['encoder_b2']))
    return layer2

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), bias['decoder_b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['decoder_h2']), bias['decoder_b2']))
    return layer2


# 输出的节点
encoder_c=encoder(x)
pred=decoder(encoder_c)

# 定义loss
cost=tf.reduce_mean(tf.pow(y-pred,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 训练参数
train_epochs=20
batch_size=256
display_step=5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch=int(mnist.train.num_examples/batch_size)
    for epoch in range(train_epochs):
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x})

        if epoch%display_step ==0:
            print("Epoch:",'{:04d}'.format(epoch+1),"cost=","{:.9f}".format(c))
    print("Done!")

    # 看预测和实际的值是否相等
    # print(pred)
    # tf.argmax就是返回最大的那个数值所在的下标,也就是返回概率最大的相应坐标是否相同，相同则为预测相同，不同则为预测不同
    # tf.equal
    correct_prediction=tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1))
    # 计算错误率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=np.float32))
    #
    # 如果t是一个tf.Tensor对象，则tf.Tensor.eval是tf.Session.run的缩写
    # （其中sess是当前的tf.get_default_session。下面的两个代码片段是等价的：
    #  sess=tf.Session()
    #  c=tf.constant(5.0)
    #  print(sess.run(c))
    #
    #  c=tf.constant(5.0)
    #  with tf.Session():
    #      print(c.eval())
    # 在第二个示例中，会话充当上下文管理器，其作用是将其安装为with块的生命周期的
    # 默认会话。 上下文管理器方法可以为简单用例（比如单元测试）提供更简洁的代码; 如果您的
    # 代码处理多个图形和会话，则可以更直接地对Session.run（）进行显式调用。
    print("Accuracy:",1-accuracy.eval({x:mnist.test.images,y:mnist.test.images}))

    show_nums=10
    # 可视化结果
    show_num = 10
    # 重构图形
    reconstruction = sess.run(
        pred, feed_dict={x: mnist.test.images[:show_num]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
    plt.show()