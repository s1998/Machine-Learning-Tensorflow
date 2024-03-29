import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric

learning_rate = 0.9
n_epochs = 10
batch_size = 10
display_step = 1
hidden_units = 100
num_classes = 549

def get_data(min_no_of_seq = 200):
	file_path = './data/db_' + str(min_no_of_seq) + '_pickle'
	# file_path = './data/db_100_pickle'
	# file_path = './data/db_50_pickle'
	file_ip = open(file_path, 'rb')
	data = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/amino_acid_map_pickle'
	file_ip = open(file_path, 'rb')
	amino_acid_map = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/families_map_pickle'
	file_ip = open(file_path, 'rb')
	families_map = pickle.load(file_ip)
	file_ip.close()

	data_train = []
	data_cv = []
	data_test = []

	# Process the data :
	# 1. Convert the data into integers.
	# 2. Now we have to divide the data
	#    into train-cv-test with 70-15-15.
	#    Do the processing class wise to 
	#    have stratification.
	# 3. Final output : 
	#    data_train
	#    data_test
	#    data_cv
	#    
	#    Each one is a list and every item in list
	#    is a mapping seq_int to fam_int

	total_no_of_seq = 0

	for k in data.keys():
		data_fam = data[k]
		shuffle(data_fam)
		fam_int = families_map[k]
		leave_the_family = False
		for entry_no in range(len(data_fam)):
			seq = data_fam[entry_no][1]
			if(len(seq) >= 1000):
				leave_the_family = True
		if(leave_the_family):
			continue
		total_no_of_seq += len(data_fam)
		for i in range(len(data_fam)):
			seq = data_fam[i][1]
			fam = data_fam[i][2]
			seq_int = []
			for j in range(len(seq)):
				seq_int.append(amino_acid_map[seq[j]])
			temp = [seq_int, fam_int]
			if(i >= 0.85*len(data_fam)):
				data_test.append(temp)
			elif(i >= 0.70*len(data_fam)):
				data_cv.append(temp)
			else:
				data_train.append(temp) 

	# for data in data_train:
	# 	print(data)
	# 	debug = input()
	print("Total, train, cv, test examples : ", total_no_of_seq,
		len(data_train), len(data_cv), len(data_test))

	return data_train, data_test, data_cv

class RnnForPfcModelOne:
	def __init__(self, 
		num_classes = 549, 
		hidden_units=100, 
		learning_rate=0.01):

		self.seq_length = tf.placeholder(tf.uint8, [None])
	
		self.x_input = tf.placeholder(tf.uint8, [None, None], name = 'x_ip')
		# batch_size * no_of_time_steps
		self.x_input_o = tf.one_hot(indices = self.x_input, 
			depth = 21,
			on_value = 1.0,
			off_value = 0.0,
			axis = -1)
		# batch_size * no_of_time_steps * 21
		self.y_input = tf.placeholder(tf.uint8, [None], name = 'y_ip')
		self.y_input_o = tf.one_hot(indices = self.y_input, 
									depth = num_classes+1,
									on_value = 1.0,
									off_value = 0.0,
									axis = -1)
		self.weights = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		self.biases = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
										   forget_bias = 1.0,
										   activation = tf.tanh)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
													  self.x_input_o,
													  sequence_length = self.seq_length,
													  dtype = tf.float32)
		self.outputs_t = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		self.y_predicted_o = tf.matmul(self.outputs_t, self.weights) + self.biases
		self.y_predicted = tf.argmax(self.y_predicted_o, 1)
		self.loss = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted_o, labels=self.y_input_o))

		# define optimizer and trainer
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.trainer = self.optimizer.minimize(self.loss)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

		self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted_o, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))
		self.summary_writer = tf.summary.FileWriter('./data/graph/', graph = self.sess.graph)
	
	def predict(self, x, y, seq_length):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})
		print(result)
		result = np.reshape(result, [-1])
		print("Converted to : \n", result)
		return result

	def optimize(self, x, y, seq_length):
		self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})

	def cross_validate(self, x, y, seq_length):
		result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length})
		return result

	def close_summary_writer(self):
		self.summary_writer.close()

	def get_loss(self, x, y, seq_length):
		result = self.sess.run(self.loss, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length})
		return result

data_train, data_test, data_cv = get_data(200)
# print(len(data_train), (len(data_test)), (len(data_cv)))

model = RnnForPfcModelOne()

def train_on_train_data():
	no_of_batches = len(data_train) // batch_size
	shuffle(data_train)
	for batch_no in range(5):
		print("Iteration number, batch number : ", epoch, batch_no)
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_train[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
			print(x)
			print(y)
			debug = input()
		x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
		y = np.array(y)
		model.optimize(x_padded, y, seq_length)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		print("Training data accuracy : ", accuracy_known)
		print("Training data loss     : ", model.get_loss(x_padded, y, seq_length))

def res_on_train_data():
	no_of_batches = len(data_train) // batch_size
	shuffle(data_train)
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(5):
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_train[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
		for y_ in y:
			y_actual.append(y_)
			print("Appending in y_actual", y_)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		for y_ in y_predicted_:
			print("Appending in y_preicted", y_)
			y_predicted.append(y_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	print("Accuracy on train data : ", accuracy)
	print (c_metric(y_actual, y_predicted))


def res_on_test_data():
	no_of_batches = len(data_test) // batch_size
	shuffle(data_test)
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(5):
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_test[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
		for y_ in y:
			y_actual.append(y_)
			print("Appending in y_actual", y_)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		for y_ in y_predicted_:
			print("Appending in y_preicted", y_)
			y_predicted.append(y_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	print("Accuracy on train data : ", accuracy)
	print (c_metric(y_actual, y_predicted))


def res_on_cv_data():
	no_of_batches = len(data_cv) // batch_size
	shuffle(data_cv)
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(5):
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_cv[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
		for y_ in y:
			y_actual.append(y_)
			print("Appending in y_actual", y_)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		for y_ in y_predicted_:
			print("Appending in y_preicted", y_)
			y_predicted.append(y_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	print("Accuracy on train data : ", accuracy)
	print (c_metric(y_actual, y_predicted))

# x_1 = [0,1,2,1,1,1,2]
# x_2 = [4,4,2,1,5,6,7]
# print("x_1", x_1)
# print("x_2", x_2)
# print(c_metric(x_1, x_2))
# x_1 [0, 1, 2, 1, 1, 1, 2]
# x_2 [4, 4, 2, 1, 5, 6, 7]
# /usr/local/lib/python3.4/dist-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#   'precision', 'predicted', average, warn_for)
# /usr/local/lib/python3.4/dist-packages/sklearn/metrics/classification.py:1115: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
#   'recall', 'true', average, warn_for)
#              precision    recall  f1-score   support

#           0       0.00      0.00      0.00         1
#           1       1.00      0.25      0.40         4
#           2       1.00      0.50      0.67         2
#           4       0.00      0.00      0.00         0
#           5       0.00      0.00      0.00         0
#           6       0.00      0.00      0.00         0
#           7       0.00      0.00      0.00         0

# avg / total       0.86      0.29      0.42         7


for epoch in range(n_epochs):
	train_on_train_data()
	res_on_train_data()
	res_on_test_data()
	res_on_cv_data()

model.close_summary_writer()
