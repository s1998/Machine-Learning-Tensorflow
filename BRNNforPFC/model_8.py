		# self.weights_f = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		# self.weights_p = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		# self.weights_h = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		# self.biases_f = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		# self.biases_p = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		# self.biases_h = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		# self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
		# 								   forget_bias = 1.0,
		# 								   activation = tf.tanh)
		# self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
		# 											  self.x_input,
		# 											  sequence_length = self.seq_length,
		# 											  dtype = tf.float32)
		
		# self.outputs_f = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		# self.outputs_maxpooled = tf.reduce_max(self.outputs, axis = 1)
		# self.outputs_p = tf.reshape(self.outputs_maxpooled, [-1, hidden_units])
		# self.h_predicted = (tf.matmul(self.outputs_f, self.weights_f) + self.biases_f 
  #  						   + tf.matmul(self.outputs_p, self.weights_p) + self.biases_p)
		# self.y_predicted = tf.matmul(tf.nn.sigmoid(self.h_predicted), self.weights_h) + self.biases_h

		
