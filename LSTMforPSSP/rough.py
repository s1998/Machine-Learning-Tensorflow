import tensorflow as tf

y = tf.placeholder(tf.int64, [2,3])
y_o = tf.one_hot(indices = y,
	depth = 4,
	axis = -1)
temp = [[1, 2, -1], [3, -1, 1]]
y_dict = {y:temp}
weights = tf.Variable(
	tf.random_uniform(shape=[2, 20, 4], 
		maxval=1, 
		dtype=tf.float64))
weights_p_l = [] 

for i in range(16):
	weights_p_l.append(tf.reduce_max(weights[:,i:i+5,:], axis = 1))
weights_p = tf.stack(weights_p_l, axis = 1)
weights_c = tf.slice(weights, [0, 1, 0], [2, 4, 4])

weights_r = tf.reshape(weights, [2, 20, 4, 1])
weights_pool = tf.nn.max_pool(weights_r, ksize = [1, 5, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
print(weights_pool.shape)
weights_pool_r = tf.reshape(weights_pool, [2, 16, 4])


# print(sess.run(y, feed_dict=y_dict)) 
# print(sess.run(y_o, feed_dict=y_dict)) 
# w, w_p, w_c, w_pool_r = (sess.run([weights, weights_p, weights_c, weights_pool_r], feed_dict=y_dict)) 
# print("w \n", w, "\n")
# print("w_p \n", w_p, "\n")
# print("w_pool_r \n", w_pool_r, "\n")
# print("w_c \n", w_c, "\n")


outputs_f = tf.Variable(tf.random_uniform(shape=[128, 800, 100], maxval=1, dtype=tf.float64))
outputs_b = tf.Variable(tf.random_uniform(shape=[128, 800, 100], maxval=1, dtype=tf.float64))

outputs_f_p_tf = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_f, [128, 800, 100, 1]), 
	ksize = [1, 50, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 751, 100]
	)[:, 0:700, :]

outputs_b_p_tf = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_b, [128, 800, 100, 1]), 
	ksize = [1, 50, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 751, 100]
	)[:, 51:751, :]

outputs_f_p_tf_20 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_f[:, 30:750, :], [128, 720, 100, 1]), 
	ksize = [1, 20, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 0:700, :]

outputs_b_p_tf_20 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_b[:, 50:770, :], [128, 720, 100, 1]), 
	ksize = [1, 20, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 1:701, :]

outputs_f_p_tf_30 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_f[:, 20:750, :], [128, 730, 100, 1]), 
	ksize = [1, 30, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 0:700, :]

outputs_b_p_tf_30 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_b[:, 50:780, :], [128, 730, 100, 1]), 
	ksize = [1, 30, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 1:701, :]

outputs_f_p_tf_10 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_f[:, 40:750, :], [128, 710, 100, 1]), 
	ksize = [1, 10, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 0:700, :]

outputs_b_p_tf_10 = tf.reshape(
	tf.nn.max_pool(
	tf.reshape(outputs_b[:, 50:760, :], [128, 710, 100, 1]), 
	ksize = [1, 10, 1, 1], 
	strides = [1, 1, 1, 1], 
	padding = 'VALID'),
	[128, 701, 100]
	)[:, 1:701, :]

outputs_f_p_50_l = []
outputs_b_p_50_l = []
outputs_f_p_20_l = []
outputs_b_p_20_l = []
outputs_f_p_10_l = []
outputs_b_p_10_l = []
outputs_f_p_30_l = []
outputs_b_p_30_l = []
for i in range(700):
  outputs_f_p_50_l.append(tf.reduce_max(outputs_f[: , i:i+50, :], axis = 1))
  outputs_b_p_50_l.append(tf.reduce_max(outputs_b[: , i+51:i+101, :], axis = 1))
  outputs_f_p_20_l.append(tf.reduce_max(outputs_f[: , i+30:i+50, :], axis = 1))
  outputs_b_p_20_l.append(tf.reduce_max(outputs_b[: , i+51:i+71, :], axis = 1))
  outputs_f_p_10_l.append(tf.reduce_max(outputs_f[: , i+40:i+50, :], axis = 1))
  outputs_b_p_10_l.append(tf.reduce_max(outputs_b[: , i+51:i+61, :], axis = 1))
  outputs_f_p_30_l.append(tf.reduce_max(outputs_f[: , i+20:i+50, :], axis = 1))
  outputs_b_p_30_l.append(tf.reduce_max(outputs_b[: , i+51:i+81, :], axis = 1))

outputs_f_p_50 = tf.stack(outputs_f_p_50_l, axis = 1)
outputs_b_p_50 = tf.stack(outputs_b_p_50_l, axis = 1)
outputs_f_p_20 = tf.stack(outputs_f_p_20_l, axis = 1)
outputs_b_p_20 = tf.stack(outputs_b_p_20_l, axis = 1)
outputs_f_p_30 = tf.stack(outputs_f_p_30_l, axis = 1)
outputs_b_p_30 = tf.stack(outputs_b_p_30_l, axis = 1)
outputs_f_p_10 = tf.stack(outputs_f_p_10_l, axis = 1)
outputs_b_p_10 = tf.stack(outputs_b_p_10_l, axis = 1)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
check_1 = tf.reduce_sum(tf.cast(tf.equal(outputs_f_p_tf, outputs_f_p_50), tf.float32))
check_2 = tf.reduce_sum(tf.cast(tf.equal(outputs_b_p_tf, outputs_b_p_50), tf.float32))
check_3 = tf.reduce_sum(tf.cast(tf.equal(outputs_f_p_tf_20, outputs_f_p_20), tf.float32))
check_4 = tf.reduce_sum(tf.cast(tf.equal(outputs_b_p_tf_20, outputs_b_p_20), tf.float32))
check_5 = tf.reduce_sum(tf.cast(tf.equal(outputs_f_p_tf_10, outputs_f_p_10), tf.float32))
check_6 = tf.reduce_sum(tf.cast(tf.equal(outputs_b_p_tf_10, outputs_b_p_10), tf.float32))
check_7 = tf.reduce_sum(tf.cast(tf.equal(outputs_f_p_tf_30, outputs_f_p_30), tf.float32))
check_8 = tf.reduce_sum(tf.cast(tf.equal(outputs_b_p_tf_30, outputs_b_p_30), tf.float32))

print("Equality check f 50 : ", sess.run(check_1, feed_dict=y_dict), sess.run(check_1, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check b 50 : ", sess.run(check_2, feed_dict=y_dict), sess.run(check_2, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check f 20 : ", sess.run(check_3, feed_dict=y_dict), sess.run(check_3, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check b 20 : ", sess.run(check_4, feed_dict=y_dict), sess.run(check_4, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check f 10 : ", sess.run(check_1, feed_dict=y_dict), sess.run(check_5, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check b 10 : ", sess.run(check_2, feed_dict=y_dict), sess.run(check_6, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check f 30 : ", sess.run(check_3, feed_dict=y_dict), sess.run(check_7, feed_dict=y_dict) == 128 * 700 * 100)
print("Equality check b 30 : ", sess.run(check_4, feed_dict=y_dict), sess.run(check_8, feed_dict=y_dict) == 128 * 700 * 100)







