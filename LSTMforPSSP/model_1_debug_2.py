import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric

def get_data_train():
  file_path = './data/batch_wise_data.pkl'
  file_ip = open(file_path, 'rb')
  data_train = pickle.load(file_ip)
  file_ip.close()
  print("Data has been loaded. ")
  return data_train

class BrnnForPsspModelOne:
  def __init__(self,
  	num_classes = 8,
  	hidden_units = 100,
  	batch_size = 5):
    
    self.input_x = tf.placeholder(tf.float64, [ batch_size, 800, 100])
    self.input_y = tf.placeholder(tf.int64, [ batch_size, 800])
    self.input_msks = tf.placeholder(tf.float64, [ batch_size, 800])
    self.input_seq_len = tf.placeholder(tf.int64, [ batch_size])
    self.input_y_o = tf.one_hot(indices = self.input_y,
      depth = num_classes,
      on_value = 1.0,
      off_value = 0.0,
      axis = -1)

    self.hidden_units = tf.constant(hidden_units, dtype = tf.float64)
    # define weights and biases here (4 weights + 4 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.biases_f_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    
    self.rnn_cell_f = rnn.GRUCell(num_units = hidden_units, 
   		activation = tf.tanh)
    self.rnn_cell_b = rnn.GRUCell(num_units = hidden_units, 
 	  	activation = tf.tanh)
    self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
 		  cell_fw = self.rnn_cell_f,
 		  cell_bw = self.rnn_cell_b,
 		  inputs = self.input_x,
 		  sequence_length = self.input_seq_len,
 		  dtype = tf.float64,
 		  swap_memory = False)
    self.outputs_f = self.outputs[0]
    self.outputs_b = self.outputs[1]
    self.outputs_f_p_50_l = []
    self.outputs_b_p_50_l = []
    self.outputs_f_p_20_l = []
    self.outputs_b_p_20_l = []
    # self.outputs_f_p_10_l = []
    # self.outputs_b_p_10_l = []
    # self.outputs_f_p_30_l = []
    # self.outputs_b_p_30_l = []
    for i in range(700):
      # 50 dummies + seq + 50 dummies
      # For forward maxpooling, index i will have maxpool from i-50:i 
      # Loss due to dummies will get maske completely 
      self.outputs_f_p_50_l.append(tf.reduce_max(self.outputs_f[: , i:i+50, :],
        axis = 1))
      self.outputs_b_p_50_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+101, :],
      	axis = 1))
      self.outputs_f_p_20_l.append(tf.reduce_max(self.outputs_f[: , i+30:i+50, :],
        axis = 1))
      self.outputs_b_p_20_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+71, :],
      	axis = 1))
      # self.outputs_f_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+40:i+50, :],
      #   axis = 1))
      # self.outputs_b_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+61, :],
      #   axis = 1))
      # self.outputs_f_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+20:i+50, :],
      #   axis = 1))
      # self.outputs_b_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+81, :],
      #   axis = 1))
    self.outputs_f_p_50 = tf.stack(self.outputs_f_p_50_l, axis = 1)
    self.outputs_b_p_50 = tf.stack(self.outputs_b_p_50_l, axis = 1)
    self.outputs_f_p_20 = tf.stack(self.outputs_f_p_20_l, axis = 1)
    self.outputs_b_p_20 = tf.stack(self.outputs_b_p_20_l, axis = 1)
    # self.outputs_f_p_10 = tf.stack(self.outputs_f_p_10_l, axis = 1)
    # self.outputs_b_p_10 = tf.stack(self.outputs_b_p_10_l, axis = 1)
    # self.outputs_f_p_30 = tf.stack(self.outputs_f_p_30_l, axis = 1)
    # self.outputs_b_p_30 = tf.stack(self.outputs_b_p_30_l, axis = 1)
    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ batch_size, 700, 100])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ batch_size, 700, 100])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 100])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 100])
    self.outputs_f_p_50_r = tf.reshape(self.outputs_f_p_50, [-1, 100])
    self.outputs_b_p_50_r = tf.reshape(self.outputs_b_p_50, [-1, 100])
    self.outputs_f_p_20_r = tf.reshape(self.outputs_f_p_20, [-1, 100])
    self.outputs_b_p_20_r = tf.reshape(self.outputs_b_p_20, [-1, 100])
    # self.outputs_f_p_30_r = tf.reshape(self.outputs_f_p_30, [-1, 100])
    # self.outputs_b_p_30_r = tf.reshape(self.outputs_b_p_30, [-1, 100])
    # self.outputs_f_p_10_r = tf.reshape(self.outputs_f_p_10, [-1, 100])
    # self.outputs_b_p_10_r = tf.reshape(self.outputs_b_p_10, [-1, 100])
    
    self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c)
                       + tf.matmul(self.outputs_b_c_r, self.weight_b_c)
                       + tf.matmul(self.outputs_f_p_50_r, self.weight_f_p_50)
                       + tf.matmul(self.outputs_b_p_50_r, self.weight_b_p_50) 
                       + tf.matmul(self.outputs_f_p_20_r, self.weight_f_p_20)
                       + tf.matmul(self.outputs_b_p_20_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_f_p_10_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_b_p_10_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_f_p_30_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_b_p_30_r, self.weight_b_p_20)
                       + self.biases)

    # self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c) + self.biases_f_c
    #                    + tf.matmul(self.outputs_b_c_r, self.weight_b_c) + self.biases_b_c
    #                    + tf.matmul(self.outputs_f_p_50_r, self.weight_f_p_50) + self.biases_f_p_50
    #                    + tf.matmul(self.outputs_b_p_50_r, self.weight_b_p_50) + self.biases_b_p_50 
    #                    + tf.matmul(self.outputs_f_p_20_r, self.weight_f_p_20) + self.biases_f_p_20
    #                    + tf.matmul(self.outputs_b_p_20_r, self.weight_b_p_20) + self.biases_b_p_20)
                       # + tf.matmul(self.outputs_f_p_30_r, self.weight_f_p_30) + self.biases_f_p_30
                       # + tf.matmul(self.outputs_b_p_30_r, self.weight_b_p_30) + self.biases_b_p_30
                       # + tf.matmul(self.outputs_f_p_10_r, self.weight_f_p_10) + self.biases_f_p_10
                       # + tf.matmul(self.outputs_b_p_10_r, self.weight_b_p_10) + self.biases_b_p_10)
    # [ batch_size*700, 8] <- self.y_predicted 
    self.input_y_o_s = tf.slice(self.input_y_o, [0, 50, 0], [ batch_size, 700, 8])
    self.input_msks_s = tf.slice(self.input_msks, [0, 50], [ batch_size, 700])
    # [ batch_size, 700, 8] <- self.input_y_o_s
    self.input_y_o_r = tf.reshape(self.input_y_o_s, [-1, 8])
    self.input_msks_r = tf.reshape(self.input_msks_s, [-1, 1])
    # [ batch_size*700, 8] <- self.input_y_o_r
    self.loss_unmasked = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.input_y_o_r), [batch_size*700, 1])
    #  dim: The class dimension. Defaulted to -1 
    #  which is the last dimension.
    self.loss_masked = tf.multiply(self.loss_unmasked, self.input_msks_r)
    self.no_of_entries_unmasked = tf.reduce_sum(self.input_msks_r)
    self.loss_reduced = ( tf.reduce_sum(self.loss_masked) / self.no_of_entries_unmasked )
	
    self.get_equal_unmasked = tf.reshape(tf.equal(tf.argmax(self.input_y_o_r, 1), tf.argmax(self.y_predicted, 1)), [batch_size*700, 1])
    self.get_equal = tf.multiply(tf.cast(self.get_equal_unmasked, tf.float64), self.input_msks_r)
    self.accuracy = ( tf.reduce_sum(tf.cast(self.get_equal, tf.float64)) / self.no_of_entries_unmasked)

    # define optimizer and trainer
    self.optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    self.trainer_1 = self.optimizer_1.minimize(self.loss_reduced)

    self.optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    self.trainer_2 = self.optimizer_2.minimize(self.loss_reduced)

    self.optimizer_3 = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    self.trainer_3 = self.optimizer_3.minimize(self.loss_reduced)

    self.optimizer_mini = tf.train.AdamOptimizer(learning_rate = 1e-2)
    self.trainer_mini = self.optimizer_mini.minimize(self.loss_reduced)

    self.sess = tf.Session()
    self.init = tf.global_variables_initializer()
    self.sess.run(self.init)

  def optimize_mini(self, x, y, seq_len, msks):
    result, loss, accuracy, no_of_entries_unmasked = self.sess.run([self.trainer_mini,
		self.loss_reduced,
		self.accuracy,
		self.no_of_entries_unmasked],
		feed_dict={self.input_x:x, 
		self.input_y:y,
		self.input_seq_len:seq_len,
		self.input_msks:msks})
    return loss, accuracy, no_of_entries_unmasked

  def get_loss_and_predictions(self, x, y, seq_len, msks):
    loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = self.sess.run([
    	self.loss_unmasked,
    	self.loss_masked,
    	self.loss_reduced,
    	self.input_msks_r,
    	self.y_predicted,
    	self.input_y_o_r],
    	feed_dict = {self.input_x:x, 
		self.input_y:y,
		self.input_seq_len:seq_len,
		self.input_msks:msks})
    return loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r 

  def print_biases(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20, biases = self.sess.run([self.biases_f_c,
      self.biases_b_c,
      self.biases_f_p_50,
      self.biases_b_p_50,
      self.biases_f_p_20,
      self.biases_b_p_20,
      self.biases],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    # print("self.biases_f_c : ", f_c)
    # print("self.biases_b_c : ", b_c)
    # print("self.biases_f_p_50 : ", f_p_50)
    # print("self.biases_b_p_50 : ", b_p_50)
    # print("self.biases_f_p_20 : ", f_p_20)
    # print("self.biases_b_p_50 : ", b_p_20)
    print("self.biases : ", biases)

  def print_weights(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = self.sess.run([self.weight_f_c,
      self.weight_b_c,
      self.weight_f_p_50,
      self.weight_b_p_50,
      self.weight_f_p_20,
      self.weight_b_p_20],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    print("self.weights_f_c : ", f_c)
    print("self.weights_b_c : ", b_c)
    print("self.weights_f_p_50 : ", f_p_50)
    print("self.weights_b_p_50 : ", b_p_50)
    print("self.weights_f_p_20 : ", f_p_20)
    print("self.weights_b_p_50 : ", b_p_20)

  def get_shapes(self):
  	print("(self.loss_unmasked.shape)", self.loss_unmasked.shape)
  	print("(self.loss_masked.shape)", self.loss_masked.shape)
  	print("(self.loss_reduced.shape)", self.loss_reduced.shape)
  	print("(self.y_predicted.shape)", self.y_predicted.shape)
  	print("(self.input_y_o_r.shape)", self.input_y_o_r.shape)
  	# print(y.y_predicted.shape)
  	print("(self.input_msks_r.shape)", self.input_msks_r.shape)
  	print("(self.get_equal_unmasked.shape)", self.get_equal_unmasked.shape)
  	print("(self.get_equal.shape)", self.get_equal.shape)
  
  def get_rnn_outputs(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = self.sess.run([self.outputs_f_c_r,
      self.outputs_b_c_r,
      self.outputs_f_p_50_r,
      self.outputs_b_p_50_r,
      self.outputs_f_p_20_r,
      self.outputs_b_p_20_r],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    return f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20

def verify_accuracy(y_inp, y_pre, msk, epoch):
  total = 0
  correct = 0
  count_5 = 0
  count_5_inp = 0
  for i in range(len(y_pre)):
    if(i%700 == 699 and epoch > 25):
      print("\n\n")
    if(msk[i // 700] [i % 700 + 50] == 1):
      if(np.argmax(y_pre[i], 0) == 5):
        count_5 += 1
      if(y_inp[i // 700][i % 700 + 50] == 5):
        count_5_inp += 1
      total += 1
      if(epoch >= 25):
        print(i, np.argmax(y_pre[i], 0), y_inp[i // 700][i % 700 + 50])
      if(np.argmax(y_pre[i], 0) == y_inp[i // 700][i % 700 + 50]):
        correct += 1
  if(epoch > 25):
    debug = input()
  print("No of 5 s predicted, input", count_5, count_5/total, count_5_inp, count_5_inp/total)
  return correct/total

def get_c1_score(y_inp, y_pre, msk):
  y_predicted = []
  y_actual = []
  for i in range(len(y_pre)):
    if(msk[i // 700] [i % 700 + 50] == 1):
      y_predicted.append(np.argmax(y_pre[i], 0))
      y_actual.append(y_inp[i // 700][i % 700 + 50])
  print("F1 score results : \n", c_metric(y_actual, y_predicted))
  print("Predicted : \n", c_metric(y_predicted, y_predicted))
  

if __name__=="__main__":
  data_train = get_data_train()
  # for batch_no in range(43):
  print("Creating model...")
  model = BrnnForPsspModelOne()
  print("Model creation finished. ")
  model.get_shapes()
  n_epochs = 200
  for epoch in range(n_epochs):
    for batch_no in range(2):
      print("Epoch number and batch_no: ", epoch, batch_no)
      data = data_train[batch_no]
      x_inp = data[0]
      y_inp = data[1]
      m_inp = data[2]
      l_inp = data[3]
      x_inp = x_inp[:5]
      y_inp = y_inp[:5]
      m_inp = m_inp[:5]
      l_inp = l_inp[:5]
      # model.print_weights(x_inp, y_inp, l_inp, m_inp)
      # f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = model.get_rnn_outputs(x_inp, y_inp, l_inp, m_inp)
      # print("f_c : ", f_c)
      # print("b_c : ", b_c)
      # print("f_p_50 : ", f_p_50)
      # print("b_p_50 : ", b_p_50)
      # print("f_p_20 : ", f_p_20)
      # print("b_p_20 : ", b_p_20)

      loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = model.get_loss_and_predictions(x_inp, y_inp, l_inp, m_inp)
      print("Loss before optimizing : ", loss_reduced)
      loss, accuracy, no_of_entries_unmasked = model.optimize_mini(x_inp, y_inp, l_inp, m_inp)
      # no_of_entries_unmasked_inp = 0
      # for i in range(5):
      # 	for j in range(len(m_inp[i])):
      # 	  no_of_entries_unmasked_inp += m_inp[i][j]
      # # print(dtype(loss_unmasked), dtype(loss_masked), dtype(loss_reduced), dtype(input_msks_r))
      ans = True
      # debugging snippet
      # for i in range(3500):
      #   print(loss_unmasked[i], loss_masked[i], input_msks_r[i], m_inp[i // 700][i % 700 + 50])
      #   ans = ans and (input_msks_r[i] == m_inp[i // 700][i % 700 + 50])
      #   ans = ans and (np.argmax(input_y_o_r[i], 0) == y_inp[i // 700][i % 700 + 50] or y_inp[i // 700][i % 700 + 50] == -1)
      #   print(y_predicted[i])
      #   print(input_y_o_r[i], y_inp[i // 700][i % 700 + 50])
      #   if(ans == False):
      #     debug = input()
      #   if(i % 700 == 699):
      #     debug = input()
      print("Loss, accuracy and verification results : ", loss, accuracy, ans)
      # print("no_of_entries_unmasked, no_of_entries_unmasked_inp", no_of_entries_unmasked, no_of_entries_unmasked_inp)
      # print("Verifying accuracy : ", verify_accuracy(y_inp, y_predicted, m_inp, epoch))
      get_c1_score(y_inp, y_predicted, m_inp)
      model.print_biases(x_inp, y_inp, l_inp, m_inp)
      # model.print_weights(x_inp, y_inp, l_inp, m_inp)








"""
Epoch number and batch_no:  0 0
Loss, accuracy :  910.271072368 275.0
Epoch number and batch_no:  0 1
Loss, accuracy :  batch_size1.51474569 291.0
Epoch number and batch_no:  1 0
Loss, accuracy :  890.325852686 279.0
Epoch number and batch_no:  1 1
Loss, accuracy :  1255.00815712 303.0
Epoch number and batch_no:  2 0
Loss, accuracy :  879.144031338 291.0
Epoch number and batch_no:  2 1
Loss, accuracy :  1239.58471894 314.0
Epoch number and batch_no:  3 0
Loss, accuracy :  874.015249401 278.0
Epoch number and batch_no:  3 1
Loss, accuracy :  1231.64421255 318.0
Epoch number and batch_no:  4 0
Loss, accuracy :  872.887020292 278.0
Epoch number and batch_no:  4 1
Loss, accuracy :  1228.63596089 326.0
Epoch number and batch_no:  5 0
Loss, accuracy :  874.31197826 283.0
Epoch number and batch_no:  5 1
Loss, accuracy :  1228.8423795 324.0
Epoch number and batch_no:  6 0
Loss, accuracy :  877.411856914 284.0
Epoch number and batch_no:  6 1
Loss, accuracy :  1231.18483386 323.0
Epoch number and batch_no:  7 0
Loss, accuracy :  881.312186267 283.0
Epoch number and batch_no:  7 1
Loss, accuracy :  1234.6607407 318.0
Epoch number and batch_no:  8 0
Loss, accuracy :  885.617966309 283.0
Epoch number and batch_no:  8 1
Loss, accuracy :  1238.65908374 318.0
Epoch number and batch_no:  9 0
Loss, accuracy :  889.887106462 280.0
Epoch number and batch_no:  9 1
Loss, accuracy :  1242.66251344 315.0
"""

"""
Epoch number and batch_no:  96 0
Loss before optimizing :  1.37624805636
Loss, accuracy and verification results :  1.37624805636 0.505976095618 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.79      0.14      0.24       152
          1       0.00      0.00      0.00         4
          2       0.88      0.30      0.45       224
          3       0.69      0.13      0.22        70
          5       0.49      1.00      0.66       381
          6       0.30      0.15      0.20        82
          7       0.22      0.18      0.20        91

avg / total       0.60      0.51      0.44      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        28
          2       1.00      1.00      1.00        77
          3       1.00      1.00      1.00        13
          5       1.00      1.00      1.00       774
          6       1.00      1.00      1.00        40
          7       1.00      1.00      1.00        72

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01553228 -0.09186028  0.07326631 -0.01915726 -0.12206331  0.05017551
 -0.03588488 -0.00974813]
Epoch number and batch_no:  96 1
Loss before optimizing :  1.24284507325
Loss, accuracy and verification results :  1.24284507325 0.516678012253 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.65      0.24      0.35       305
          1       0.20      0.09      0.13        11
          2       0.71      0.52      0.60       289
          3       0.30      0.40      0.34        82
          5       0.67      0.81      0.73       475
          6       0.44      0.17      0.24       138
          7       0.23      0.55      0.33       169

avg / total       0.58      0.52      0.51      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       111
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       212
          3       1.00      1.00      1.00       110
          5       1.00      1.00      1.00       580
          6       1.00      1.00      1.00        52
          7       1.00      1.00      1.00       399

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01764044 -0.09198615  0.07578766 -0.0206055  -0.12208426  0.04867134
 -0.03534396 -0.0128427 ]
Epoch number and batch_no:  97 0
Loss before optimizing :  1.32511782231
Loss, accuracy and verification results :  1.32511782231 0.497011952191 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.30      0.51      0.38       152
          1       0.00      0.00      0.00         4
          2       0.42      0.89      0.57       224
          3       0.66      0.41      0.51        70
          5       0.85      0.50      0.63       381
          6       0.60      0.04      0.07        82
          7       0.00      0.00      0.00        91

avg / total       0.56      0.50      0.46      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       256
          2       1.00      1.00      1.00       476
          3       1.00      1.00      1.00        44
          5       1.00      1.00      1.00       223
          6       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01710549 -0.09214627  0.07546956 -0.0220802  -0.12210449  0.04961335
 -0.034228   -0.01425385]
Epoch number and batch_no:  97 1
Loss before optimizing :  1.34948807147
Loss, accuracy and verification results :  1.34948807147 0.495575221239 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.36      0.49      0.41       305
          1       0.00      0.00      0.00        11
          2       0.48      0.73      0.58       289
          3       1.00      0.01      0.02        82
          5       0.61      0.72      0.66       475
          6       0.44      0.17      0.25       138
          7       0.00      0.00      0.00       169

avg / total       0.46      0.50      0.44      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       412
          2       1.00      1.00      1.00       440
          3       1.00      1.00      1.00         1
          5       1.00      1.00      1.00       561
          6       1.00      1.00      1.00        55

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01543615 -0.09200706  0.07428743 -0.02230088 -0.12212474  0.05062184
 -0.03360555 -0.012824  ]
Epoch number and batch_no:  98 0
Loss before optimizing :  1.12861186578
Loss, accuracy and verification results :  1.12861186578 0.581673306773 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.22      0.32       152
          1       0.00      0.00      0.00         4
          2       0.68      0.59      0.63       224
          3       0.65      0.21      0.32        70
          5       0.58      0.98      0.73       381
          6       0.35      0.37      0.36        82
          7       0.50      0.01      0.02        91

avg / total       0.58      0.58      0.52      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        57
          2       1.00      1.00      1.00       197
          3       1.00      1.00      1.00        23
          5       1.00      1.00      1.00       639
          6       1.00      1.00      1.00        86
          7       1.00      1.00      1.00         2

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01480297 -0.09183728  0.0745125  -0.02124815 -0.12214543  0.04981957
 -0.03402408 -0.01013637]
Epoch number and batch_no:  98 1
Loss before optimizing :  1.29157145131
Loss, accuracy and verification results :  1.29157145131 0.515997277059 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.13      0.22       305
          1       0.40      0.18      0.25        11
          2       0.56      0.61      0.58       289
          3       0.71      0.06      0.11        82
          5       0.56      0.92      0.70       475
          6       0.31      0.35      0.33       138
          7       0.31      0.30      0.30       169

avg / total       0.56      0.52      0.46      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        53
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       315
          3       1.00      1.00      1.00         7
          5       1.00      1.00      1.00       774
          6       1.00      1.00      1.00       154
          7       1.00      1.00      1.00       161

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01652751 -0.0917603   0.07504127 -0.01962564 -0.12216629  0.04836316
 -0.03560296 -0.00912291]
Epoch number and batch_no:  99 0
Loss before optimizing :  1.03699146106
Loss, accuracy and verification results :  1.03699146106 0.609561752988 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.34      0.44       152
          1       0.00      0.00      0.00         4
          2       0.56      0.85      0.67       224
          3       0.46      0.69      0.55        70
          5       0.86      0.76      0.81       381
          6       0.50      0.07      0.13        82
          7       0.21      0.30      0.25        91

avg / total       0.64      0.61      0.59      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        81
          2       1.00      1.00      1.00       342
          3       1.00      1.00      1.00       105
          5       1.00      1.00      1.00       336
          6       1.00      1.00      1.00        12
          7       1.00      1.00      1.00       128

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01834752 -0.09179094  0.07472362 -0.01993384 -0.12218665  0.04826461
 -0.03622316 -0.01030711]
Epoch number and batch_no:  99 1
Loss before optimizing :  1.2505041845
Loss, accuracy and verification results :  1.2505041845 0.489448604493 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.31      0.70      0.43       305
          1       0.40      0.18      0.25        11
          2       0.57      0.68      0.62       289
          3       0.38      0.18      0.25        82
          5       0.78      0.60      0.68       475
          6       0.00      0.00      0.00       138
          7       0.30      0.04      0.06       169

avg / total       0.49      0.49      0.45      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       690
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       347
          3       1.00      1.00      1.00        40
          5       1.00      1.00      1.00       366
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00        20

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01729476 -0.09207176  0.07439135 -0.02128846 -0.12220774  0.04928387
 -0.03523451 -0.01097222]
"""

























