import numpy as np
import tensorflow as tf


def test(argv):
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300
    # print(x_data)

    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(0, 201):
        result = sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))


def get_weight(shape, lambda_):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_)(var))
    return var


def main(argv):
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    # x=np.float32(np.random.rand(100,2))
    # y_=np.float32(np.random.rand(100,1))
    batch_size = 8
    layer_dimension = [2, 10, 10, 10, 1]
    n_layers = len(layer_dimension)
    cur_layer = x
    in_dimension = layer_dimension[0]

    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight = get_weight([in_dimension, out_dimension], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        in_dimension = layer_dimension[i]

    mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
    loss = tf.add_to_collection('losses', mse_loss)
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    sess.run(mse_loss,feed_dict={x: [[0.7, 0.9]],y_:[[1]]})



def test1():
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))
    x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(y, feed_dict={x: [[0.7, 0.9]]})


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)
    # tf.app.run(test)

    import tensorflow as tf

    labels = [[0.2, 0.3, 0.5],
              [0.1, 0.6, 0.3]]
    logits = [[4, 1, -2],
              [0.1, 1, 3]]

    logits_scaled = tf.nn.softmax(logits)
    result = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # result = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

    with tf.Session() as sess:
        print(sess.run(logits_scaled))
        print(sess.run(result))

    # test1()
# loss=tf.add_n(tf.get_collection('losses'))
