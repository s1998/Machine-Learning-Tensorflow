import tensorflow as tf
from model_5 import RnnForPfcModelFive

w = tf.Variable(tf.random_uniform(shape=[2, 3, 4], maxval=1))
init = tf.global_variables_initializer()

with tf.Session() as sess:
	print("Running tf.reshape : ")
	print("Initailizing : ", sess.run(init))
	print(sess.run(w))
	print(sess.run(tf.reshape(w, [-1, 4])))
	print(sess.run(tf.reshape(w[:, -1, :], [-1, 4])))



