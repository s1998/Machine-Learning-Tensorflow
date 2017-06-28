import tensorflow as tf
import numpy as np
# from model_5 import RnnForPfcModelFive

# w = tf.Variable(tf.random_uniform(shape=[2, 3, 4], maxval=1))
# init = tf.global_variables_initializer()
# x = tf.placeholder(tf.float32, [None])
# x_inv = tf.div(x*0+1, x)

# with tf.Session() as sess:
# 	print("Running tf.reshape : ")
# 	print("Initailizing : ", sess.run(init))
# 	print(sess.run(w))
# 	print(sess.run(tf.reshape(w, [-1, 4])))
# 	print(sess.run(tf.reshape(w[:, -1, :], [-1, 4])))
# 	print(sess.run(x_inv, feed_dict = {x : [1,2,3,4]}))

x_1 = [1,2,3,4,5,6]
x_2 = [2,2,4,4,6,6]
x_2 = np.array(x_2)
x_3 = []
x_3.extend(x_2)
print(zip(x_1, x_3))
print([i for i,j in zip(x_1, x_3)])
print([j for i,j in zip(x_1, x_3)])
print([i for i,j in zip(x_1, x_3) if i == j])
print(len([i for i,j in zip(x_1, x_3) if i == j]))

