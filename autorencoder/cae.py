#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-13
@Author  : fay
@Email   : fay625@sina.cn
@File    : cae.py
@Software: PyCharm
"""
"""
https://blog.csdn.net/coderpai/article/details/80412549
https://blog.csdn.net/g8015108/article/details/60322338

VAE 变分自编码器
https://www.leiphone.com/news/201707/7PjloKpQx1ljroGw.html
"""
import tensorflow as tf
from keras import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data",one_hot=True)
learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
epochs = 100
batch_size = 200
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        # Get images from the batch
        imgs = batch[0].reshape((-1, 28, 28, 1))

        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})
    print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

# def getModel():
#     input_img = Input(shape=(48, 48, 1))
#     x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
#     x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
#     x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
#     x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
#     x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
#     encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
#     #6x6x32 -- bottleneck
#     x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
#     x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
#     x = UpSampling2D((2, 2), dim_ordering='tf')(x)
#     x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
#     decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
#
#     #Create model
#     autoencoder = Model(input_img, decoded)
#     return autoencoder
#
# # Trains the model for 10 epochs
# def trainModel():
#     # Load dataset
#     print("Loading dataset...")
#     x_train_gray, x_train, x_test_gray, x_test = getDataset()
#
#     # Create model description
#     print("Creating model...")
#     model = getModel()
#     model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
#
#     # Train model
#     print("Training model...")
#     model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
#
#     # Evaluate loaded model on test data
#     print("Evaluating model...")
#     score = model.evaluate(x_train_gray, x_train, verbose=0)
#     print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
#
#     # Serialize model to JSON
#     print("Saving model...")
#     model_json = model.to_json()
#     with open("model.json", "w") as json_file:
#         json_file.write(model_json)
#
#     # Serialize weights to HDF5
#     print("Saving weights...")
#     model.save_weights("model.h5")



