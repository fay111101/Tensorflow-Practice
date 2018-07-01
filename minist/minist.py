import minist.input_data as input_data
import tensorflow as tf

sess=tf.InteractiveSession
mnist=input_data.read_data_sets('MINIST_data',one_hot=True)
x=tf.placeholder("float",shape=[None,784])
y=tf.placeholder("float",shape=[None,10])
W=tf.Variable(tf.zeros[784,10])
b=tf.Variable(tf.zeros[10])

