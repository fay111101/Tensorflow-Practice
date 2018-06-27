import tensorflow as tf
import numpy as np
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.100,0.200],x_data)+0.300
# print(x_data)

b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,x_data)+b
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for step in range(0,201):
    result=sess.run(train)
    if step %20 ==0:
        print(step,sess.run(W),sess.run(b))

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))
