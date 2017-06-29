import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric

def get_data_glove(min_no_of_seq = 200):
	file_path = './data/data_train_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_train = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_cv_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_cv = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_test_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_test = pickle.load(file_ip)
	file_ip.close()

	return data_train, data_test, data_cv

class RnnForPfcModelSeven:
	def __init__(self, 
		num_classes = 549, 
		hidden_units=100, 
		learning_rate=0.001):

		self.seq_length = tf.placeholder(tf.int64, [None])
		self.freq = tf.placeholder(tf.float64, [None])
		self.freq_inv = tf.div(self.freq * 0 + 1, self.freq)
		self.learning_rate = learning_rate	
		self.x_input = tf.placeholder(tf.float64, [None, None, 100], name = 'x_ip')
		# batch_size * no_of_time_steps * 100
		self.y_input = tf.placeholder(tf.int64, [None], name = 'y_ip')
		self.y_input_o = tf.one_hot(indices = self.y_input, 
									depth = num_classes+1,
									on_value = 1.0,
									off_value = 0.0,
									axis = -1)
		
		# 
		self.weights_f = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1, dtype=tf.float64), dtype=tf.float64)
		self.weights_p = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1, dtype=tf.float64), dtype=tf.float64)
		self.biases_f = tf.Variable(tf.random_uniform(shape=[num_classes+1], dtype=tf.float64), dtype=tf.float64)
		self.biases_p = tf.Variable(tf.random_uniform(shape=[num_classes+1], dtype=tf.float64), dtype=tf.float64)
		self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
										   forget_bias = 1.0,
										   activation = tf.sigmoid)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
													  self.x_input,
													  sequence_length = self.seq_length,
													  dtype = tf.float64)
		
		self.outputs_f = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		self.outputs_maxpooled = tf.reduce_max(self.outputs, axis = 1)
		self.outputs_p = tf.reshape(self.outputs_maxpooled, [-1, hidden_units])
		self.y_predicted = (tf.matmul(self.outputs_f, self.weights_f) + self.biases_f 
   						   + tf.matmul(self.outputs_p, self.weights_p) + self.biases_p)
		# 

		self.loss_unweighted = (
					tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input_o))
		self.loss_weighted = tf.multiply(self.loss_unweighted, self.freq_inv)
		self.loss_reduced = tf.reduce_mean(self.loss_weighted) * 498

		# define optimizer and trainer
		self.optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
		self.trainer_1 = self.optimizer_1.minimize(self.loss_reduced)

		self.optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
		self.trainer_2 = self.optimizer_2.minimize(self.loss_reduced)

		self.optimizer_3 = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		self.trainer_3 = self.optimizer_3.minimize(self.loss_reduced)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

		self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float64))

	def predict(self, x, y, seq_length, freq):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})
		result = np.argmax(result, axis=1)
		result = np.reshape(result, [-1])
		return result

	def optimize_1(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_1, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_2(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_2, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_3(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_3, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})
	
	def cross_validate(self, x, y, seq_length, freq):
		result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length, self.freq:freq})
		return result

	def get_loss(self, x, y, seq_length, freq):
		result = self.sess.run(self.loss_reduced, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length, self.freq:freq})
		return result

use_optimizer = 1

def train_on_train_data(epoch, model, data_train, data_test):
	global use_optimizer
	no_of_batches = len(data_train.keys())
	# for batch_no in range(70):
	for batch_no in range(no_of_batches-1, -1, -1):
	# for batch_no in [167, 160, 120, 80, 40, 10]:
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_train[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = data_batch[2]
		seq_length = data_batch[3]
		x = np.array(x)
		y_n = np.array(y)
		freq = np.array(freq)
		if use_optimizer == 1:
			model.optimize_1(x, y_n, seq_length, freq)
		elif use_optimizer == 2:
			model.optimize_2(x, y_n, seq_length, freq)
		elif use_optimizer == 3:
			model.optimize_3(x, y_n, seq_length, freq)
		accuracy = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted = y_predicted_
		# print (c_metric(y, y_predicted))
		print("Training data accuracy : ", accuracy)
		print("Training data loss     : ", model.get_loss(x, y, seq_length, freq))
		if(use_optimizer == 1 and accuracy > 0.85):
			use_optimizer = 2
		if(use_optimizer == 2 and accuracy > 0.92):
			use_optimizer = 3

def res_on_train_data(model, data_train):
	no_of_batches = len(data_train.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
	# for batch_no in range(70):
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_train[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		# freq = data_batch[2]
		freq = [140] * batch_size
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on train data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

def res_on_test_data(model, data_test):
	no_of_batches = len(data_test.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_test[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = data_batch[2]
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on test data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

def res_on_cv_data(model, data_cv):
	no_of_batches = len(data_cv.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_cv[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = data_batch[2]
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on cv data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

if __name__=="__main__":
	# learning_rate = 0.01
	n_epochs = 10
	batch_size = 1000
	hidden_units = 100
	num_classes = 498
	data_train, data_test, data_cv = get_data_glove(200)
	model = RnnForPfcModelSeven(num_classes = 498, hidden_units=100, learning_rate=0.001)
	for epoch in range(n_epochs):
		train_on_train_data(epoch, model, data_train, data_test)
		# res_on_train_data(model, data_train)
		res_on_test_data(model, data_test)
		res_on_cv_data(model, data_cv)
