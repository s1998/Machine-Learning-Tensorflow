import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
import time
from sklearn.metrics import classification_report as c_metric
import os
import sys
tf.logging.set_verbosity(tf.logging.ERROR)

def save_obj(obj,filename,overwrite=1):
  if(not overwrite and os.path.exists(filename)):
    return
  with open(filename,'wb') as f:
    pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    print("File saved to " + filename)

def load_obj(filename):
  with open(filename, 'rb') as f:
    obj = pickle.load(f)
    print("File loaded from " + filename)
    return obj

def get_data_train():
  file_path = './data/batch_wise_data_128.pkl'
  file_path_1 = './data/batch_wise_data_128_test.pkl'
  p=time.time()
  with open(file_path, 'rb') as file_ip:
    data_train = pickle.load(file_ip)
  with open(file_path_1, 'rb') as file_ip:
    data_test = pickle.load(file_ip)
  print("Data has been loaded in %d seconds" % (time.time()-p) )
  return data_train, data_test

class BrnnForPsspModelOne:
  def __init__(self,model_path,load_model_filename,curr_model_filename,
    num_classes = 8,
    hidden_units = 100,
    batch_size = 128):
    print("Initializing model..")
    p=time.time()

    self.input_x = tf.placeholder(tf.float32, [ batch_size, 800, 122])
    self.input_y = tf.placeholder(tf.uint8, [ batch_size, 800]) # Int 8 will be sufficient for just 8 classes.
    self.input_msks = tf.placeholder(tf.float32, [ batch_size, 800])
    self.input_seq_len = tf.placeholder(tf.int32, [ batch_size])
    self.input_y_o = tf.one_hot(indices = self.input_y,
      depth = num_classes,
      on_value = 1.0,
      off_value = 0.0,
      axis = -1)

    self.hidden_units = tf.constant(hidden_units, dtype = tf.float32)
    
    # define weights and biases here (8 weights + 1 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    
    self.rnn_cell_f = rnn.GRUCell(num_units = hidden_units, 
      activation = tf.tanh)
    self.rnn_cell_b = rnn.GRUCell(num_units = hidden_units, 
      activation = tf.tanh)
    self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
      cell_fw = self.rnn_cell_f,
      cell_bw = self.rnn_cell_b,
      inputs = self.input_x,
      sequence_length = self.input_seq_len,
      dtype = tf.float32,
      swap_memory = False)
    self.outputs_f = self.outputs[0]
    self.outputs_b = self.outputs[1]

    self.outputs_f_p_50 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_f, [batch_size, 800, 100, 1]), 
                              ksize = [1, 50, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 751, 100]
                            )[:, 0:700, :]
    self.outputs_b_p_50 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_b, [batch_size, 800, 100, 1]), 
                              ksize = [1, 50, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 751, 100]
                            )[:, 51:751, :]
    self.outputs_f_p_20 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_f[:, 30:750, :], [batch_size, 720, 100, 1]), 
                              ksize = [1, 20, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 701, 100]
                            )[:, 0:700, :]
    self.outputs_b_p_20 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_b[:, 50:770, :], [batch_size, 720, 100, 1]), 
                              ksize = [1, 20, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 701, 100]
                            )[:, 1:701, :]
    self.outputs_f_p_30 = tf.reshape(
                                tf.nn.max_pool(
                                  tf.reshape(self.outputs_f[:, 20:750, :], [batch_size, 730, 100, 1]), 
                                  ksize = [1, 30, 1, 1], 
                                  strides = [1, 1, 1, 1], 
                                  padding = 'VALID'),
                                [batch_size, 701, 100]
                                )[:, 0:700, :]
    self.outputs_b_p_30 = tf.reshape(
                                tf.nn.max_pool(
                                  tf.reshape(self.outputs_b[:, 50:780, :], [batch_size, 730, 100, 1]), 
                                  ksize = [1, 30, 1, 1], 
                                  strides = [1, 1, 1, 1], 
                                  padding = 'VALID'),
                                [batch_size, 701, 100]
                                )[:, 1:701, :]
    self.outputs_f_p_10 = tf.reshape(
                                tf.nn.max_pool(
                                    tf.reshape(self.outputs_f[:, 40:750, :], [batch_size, 710, 100, 1]), 
                                    ksize = [1, 10, 1, 1], 
                                    strides = [1, 1, 1, 1], 
                                    padding = 'VALID'),
                                [batch_size, 701, 100]
                                )[:, 0:700, :]
    self.outputs_b_p_10 = tf.reshape(
                              tf.nn.max_pool(
                                tf.reshape(self.outputs_b[:, 50:760, :], [batch_size, 710, 100, 1]), 
                                  ksize = [1, 10, 1, 1], 
                                  strides = [1, 1, 1, 1], 
                                  padding = 'VALID'),
                                [batch_size, 701, 100]
                                )[:, 1:701, :]
    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ batch_size, 700, 100])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ batch_size, 700, 100])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 100])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 100])
    self.outputs_f_p_50_r = tf.reshape(self.outputs_f_p_50, [-1, 100])
    self.outputs_b_p_50_r = tf.reshape(self.outputs_b_p_50, [-1, 100])
    self.outputs_f_p_20_r = tf.reshape(self.outputs_f_p_20, [-1, 100])
    self.outputs_b_p_20_r = tf.reshape(self.outputs_b_p_20, [-1, 100])
    self.outputs_f_p_30_r = tf.reshape(self.outputs_f_p_30, [-1, 100])
    self.outputs_b_p_30_r = tf.reshape(self.outputs_b_p_30, [-1, 100])
    self.outputs_f_p_10_r = tf.reshape(self.outputs_f_p_10, [-1, 100])
    self.outputs_b_p_10_r = tf.reshape(self.outputs_b_p_10, [-1, 100])
    
    self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c)
                       + tf.matmul(self.outputs_b_c_r, self.weight_b_c)
                       + tf.matmul(self.outputs_f_p_50_r, self.weight_f_p_50)
                       + tf.matmul(self.outputs_b_p_50_r, self.weight_b_p_50) 
                       + tf.matmul(self.outputs_f_p_20_r, self.weight_f_p_20)
                       + tf.matmul(self.outputs_b_p_20_r, self.weight_b_p_20)
                       + tf.matmul(self.outputs_f_p_10_r, self.weight_f_p_10)
                       + tf.matmul(self.outputs_b_p_10_r, self.weight_b_p_10)
                       + tf.matmul(self.outputs_f_p_30_r, self.weight_f_p_30)
                       + tf.matmul(self.outputs_b_p_30_r, self.weight_b_p_30)
                       + self.biases)

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
    self.get_equal = tf.multiply(tf.cast(self.get_equal_unmasked, tf.float32), self.input_msks_r)
    self.accuracy = ( tf.reduce_sum(tf.cast(self.get_equal, tf.float32)) / self.no_of_entries_unmasked)

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
    # 'Saver' op to save and restore all the variables
    self.saver = tf.train.Saver()

    # Restore model weights from previously saved model
    self.load_file_path = model_path+load_model_filename
    self.curr_file_path = model_path+curr_model_filename
    
    print("Model Initialized in %d seconds " % (time.time()-p))
    if os.path.exists(self.load_file_path):
      print("Restoring model...")
      p=time.time()
      self.sess.run(self.init)
      saver.restore(self.sess, self.load_file_path)
      print("Model restored from file: %s in %d seconds " % (save_path,time.time()-p))
    else:
      print("Load file DNE at "+load_model_filename+", Preparing new model...")
      #just make dir if DNE
      if not os.path.exists(model_path):
        print("created DIR "+model_path)
        os.makedirs(model_path)
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

  def get_loss_and_accuracy(self, x, y, seq_len, msks):
    loss, accuracy, no_of_entries_unmasked = self.sess.run([
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
    biases = self.sess.run([
      self.biases],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    print("self.biases : ", np.array_repr(np.array(biases)).replace('\n', ''))

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
    print("self.loss_unmasked.shape", self.loss_unmasked.shape)
    print("self.loss_masked.shape", self.loss_masked.shape)
    print("self.loss_reduced.shape", self.loss_reduced.shape)
    print("self.y_predicted.shape", self.y_predicted.shape)
    print("self.input_y_o_r.shape", self.input_y_o_r.shape)
    # print(y.y_predicted.shape)
    print("self.input_msks_r.shape", self.input_msks_r.shape)
    print("self.get_equal_unmasked.shape", self.get_equal_unmasked.shape)
    print("self.get_equal.shape", self.get_equal.shape)
    
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
  #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##
  model_path = "/tmp/LSTMmodels/"
  remake_chkpt=True
  args=sys.argv
  file_index=1
  if(len(args)>1):
    remake_chkpt = int(args[1])==0
    file_index= int(args[1])

  model_filenames_pkl = model_path+'model_filenames_pkl.pkl'
  epoch_wise_accs_pkl = model_path+'epoch_wise_accs_pkl.pkl'
  epoch_wise_loss_pkl = model_path+'epoch_wise_loss_pkl.pkl'
  start_time = time.strftime("%b%d_%H:%M%p") #by default takes current time
  curr_model_filename = "model_started_"+start_time+"_.ckpt"
  
  if(os.path.exists(model_filenames_pkl)):
    model_filenames = load_obj(model_filenames_pkl) #next time
  else:
    model_filenames=[curr_model_filename] #first time.

  if(remake_chkpt):
    print("Adding new checkpoint file")
    load_model_filename = curr_model_filename
  else:
    if( file_index > len(model_filenames) ):
      raise ValueError("Invalid file index. Avl checkpoints are : ",model_filenames)
    load_model_filename = model_filenames[-1* file_index]
    print("Loading model from file ",load_model_filename)
  #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##

  # Restore will happen from inside the class
  model = BrnnForPsspModelOne(model_path,load_model_filename,curr_model_filename)

  data_train, data_test = get_data_train()
  # for batch_no in range(43):
  model.get_shapes()
  batch_size = 128
  n_epochs = 50
  num_batches= 5534 // batch_size
  num_batches_test= 513 // batch_size
  
  # Want = Accuracies of each epochs printed into a file.
  epoch_wise_accs = []
  epoch_wise_loss = []

  for epoch in range(n_epochs):
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    for batch_no in range(num_batches):
      print("Epoch number and batch_no: ", epoch, batch_no)
      data = data_train[batch_no]
      x_inp = data[0]
      y_inp = data[1]
      m_inp = data[2]
      l_inp = data[3]
      
      loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = model.get_loss_and_predictions(x_inp, y_inp, l_inp, m_inp)
      print("Loss before optimizing : ", loss_reduced)
      loss, accuracy, no_of_entries_unmasked = model.optimize_mini(x_inp, y_inp, l_inp, m_inp)
      print("Loss, accuracy and verification results : ", loss, accuracy)
      get_c1_score(y_inp, y_predicted, m_inp)
      model.print_biases(x_inp, y_inp, l_inp, m_inp)
      acc_train.append(accuracy)
      loss_train.append(loss)
    for batch_no in range(num_batches_test):
      print("Epoch number and testing batch number : ", epoch, batch_no)
      data = data_test[batch_no]
      x_inp = data[0]
      y_inp = data[1]
      m_inp = data[2]
      l_inp = data[3]
      loss, accuracy, no_of_entries_unmasked = model.get_loss_and_accuracy(x_inp, y_inp, l_inp, m_inp)
      print("Loss, accuracy and verification results : ", loss, accuracy)
      get_c1_score(y_inp, y_predicted, m_inp)
      acc_test.append(accuracy)
      loss_test.append(loss)
    
   acc_train_avg = 0
    loss_train_avg = 0
    for i in range(len(acc_train)):
      acc_train_avg += acc_train[i]
      loss_train_avg += loss_train[i]
    acc_train_avg = acc_train_avg / len(acc_train)
    loss_train_avg = loss_train_avg / len(loss_train)

    acc_test_avg = 0
    loss_test_avg = 0
    for i in range(len(acc_test)):
      acc_test_avg += acc_test[i]
      loss_test_avg += loss_test[i]
    acc_test_avg = acc_test_avg / len(acc_test)
    loss_test_avg = loss_test_avg / len(loss_test)

    print("\n\n\n")
    print("Epoch number and results on train data : ", i, acc_train_avg, loss_train_avg)
    print("Epoch number and results on test data  : ", i, acc_test_avg, loss_test_avg)
    epoch_wise_accs.append([acc_train_avg, acc_test_avg])
    epoch_wise_loss.append([loss_train_avg, loss_test_avg])
    
    print('')
    # Save model weights to disk
    p=time.time()
    save_path = model.saver.save(model.sess, model.curr_file_path,global_step=epoch)
    model_filenames.append(save_path.split('/')[-1])
    print("Epoch %d : Model saved in file: %s in %d seconds " % (epoch, save_path,time.time()-p))
    save_obj(model_filenames,model_filenames_pkl,overwrite=1)
    save_obj(epoch_wise_accs,epoch_wise_accs_pkl,overwrite=1)
    save_obj(epoch_wise_loss,epoch_wise_loss_pkl,overwrite=1)
    print("Current saved checkpoints : ",model_filenames)
    print('')
    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##

"""

Epoch number and batch_no:  24 0
Loss before optimizing :  0.772665
Loss, accuracy and verification results :  0.772665 0.721093
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.69      0.61      5354
          1       0.75      0.02      0.04       272
          2       0.82      0.82      0.82      5867
          3       0.53      0.16      0.24      1128
          4       0.00      0.00      0.00        12
          5       0.88      0.92      0.90     10020
          6       0.42      0.27      0.33      2371
          7       0.54      0.56      0.55      3118

avg / total       0.71      0.72      0.71     28142

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6637
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00      5833
          3       1.00      1.00      1.00       330
          5       1.00      1.00      1.00     10487
          6       1.00      1.00      1.00      1565
          7       1.00      1.00      1.00      3282

avg / total       1.00      1.00      1.00     28142

self.biases :  array([[ 0.07871722, -0.10666627,  0.0312885 , -0.09511936, -0.1176827 , 0.07463975, -0.05208393, -0.02114869]], dtype=float32)
Epoch number and batch_no:  24 1
Loss before optimizing :  0.742551
Loss, accuracy and verification results :  0.742551 0.730443
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.66      0.61      5407
          1       0.50      0.01      0.02       269
          2       0.80      0.86      0.83      6449
          3       0.69      0.10      0.18       975
          5       0.87      0.93      0.90      9795
          6       0.49      0.24      0.33      2437
          7       0.54      0.58      0.56      3174

avg / total       0.72      0.73      0.71     28506

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6323
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00      6918
          3       1.00      1.00      1.00       144
          5       1.00      1.00      1.00     10496
          6       1.00      1.00      1.00      1222
          7       1.00      1.00      1.00      3397

avg / total       1.00      1.00      1.00     28506

self.biases :  array([[ 0.07836414, -0.10662069,  0.03156305, -0.09429678, -0.11771374, 0.07492013, -0.05274583, -0.02143718]], dtype=float32)
Epoch number and batch_no:  24 2
Loss before optimizing :  0.764658
Loss, accuracy and verification results :  0.764658 0.723591
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.63      0.60      5531
          1       0.45      0.02      0.03       288
          2       0.75      0.87      0.81      5868
          3       0.57      0.18      0.27      1161
          4       0.00      0.00      0.00         6
          5       0.84      0.95      0.89     10798
          6       0.47      0.17      0.24      2265
          7       0.56      0.51      0.53      3192

avg / total       0.70      0.72      0.70     29109

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5994
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00      6778
          3       1.00      1.00      1.00       366
          5       1.00      1.00      1.00     12262
          6       1.00      1.00      1.00       804
          7       1.00      1.00      1.00      2894

avg / total       1.00      1.00      1.00     29109

self.biases :  array([[ 0.07865993, -0.1066669 ,  0.03127258, -0.09252238, -0.1176962 , 0.0743373 , -0.05299836, -0.02122982]], dtype=float32)
Epoch number and batch_no:  24 3
Loss before optimizing :  0.788642
Loss, accuracy and verification results :  0.788642 0.711216
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.66      0.60      4883
          1       0.47      0.04      0.07       246
          2       0.76      0.85      0.80      5512
          3       0.35      0.40      0.37       917
          5       0.90      0.91      0.90      8123
          6       0.52      0.12      0.20      2034
          7       0.57      0.54      0.55      2857

avg / total       0.70      0.71      0.69     24572

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5939
          1       1.00      1.00      1.00        19
          2       1.00      1.00      1.00      6174
          3       1.00      1.00      1.00      1046
          5       1.00      1.00      1.00      8229
          6       1.00      1.00      1.00       484
          7       1.00      1.00      1.00      2681

avg / total       1.00      1.00      1.00     24572

self.biases :  array([[ 0.07873708, -0.10733859,  0.03075297, -0.09360427, -0.11769734, 0.07456625, -0.05227291, -0.02078923]], dtype=float32)
Epoch number and batch_no:  24 4
Loss before optimizing :  0.751583
Loss, accuracy and verification results :  0.751583 0.728656
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.69      0.61      4977
          1       0.00      0.00      0.00       250
          2       0.82      0.83      0.83      5375
          3       0.49      0.26      0.34      1020
          4       0.00      0.00      0.00        10
          5       0.89      0.92      0.91      9744
          6       0.46      0.20      0.28      2065
          7       0.52      0.56      0.54      2983

avg / total       0.71      0.73      0.71     26424

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6194
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00      5470
          3       1.00      1.00      1.00       537
          5       1.00      1.00      1.00     10114
          6       1.00      1.00      1.00       923
          7       1.00      1.00      1.00      3184

avg / total       1.00      1.00      1.00     26424

self.biases :  array([[ 0.07839358, -0.10756785,  0.03069585, -0.09488192, -0.11760361, 0.07528602, -0.05191827, -0.02099118]], dtype=float32)
Epoch number and batch_no:  24 5
Loss before optimizing :  0.783549
Loss, accuracy and verification results :  0.783549 0.714248
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.63      0.61      5185
          1       0.67      0.01      0.03       289
          2       0.81      0.83      0.82      5609
          3       0.67      0.15      0.24      1063
          4       0.00      0.00      0.00         6
          5       0.82      0.94      0.88      9113
          6       0.38      0.28      0.33      2125
          7       0.54      0.54      0.54      3154

avg / total       0.70      0.71      0.70     26544

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5377
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00      5729
          3       1.00      1.00      1.00       232
          5       1.00      1.00      1.00     10461
          6       1.00      1.00      1.00      1576
          7       1.00      1.00      1.00      3163

avg / total       1.00      1.00      1.00     26544

self.biases :  array([[ 0.07882353, -0.10705434,  0.03120721, -0.09478401, -0.11746619, 0.0750343 , -0.05282249, -0.02121253]], dtype=float32)
Epoch number and batch_no:  24 6
Loss before optimizing :  0.788518
Loss, accuracy and verification results :  0.788518 0.706991
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.65      0.60      5419
          1       0.76      0.06      0.10       286
          2       0.76      0.86      0.81      6183
          3       0.50      0.13      0.21       977
          4       0.00      0.00      0.00        13
          5       0.86      0.93      0.89      8756
          6       0.46      0.16      0.23      2308
          7       0.51      0.53      0.52      2948

avg / total       0.69      0.71      0.68     26890

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6337
          1       1.00      1.00      1.00        21
          2       1.00      1.00      1.00      6974
          3       1.00      1.00      1.00       257
          5       1.00      1.00      1.00      9440
          6       1.00      1.00      1.00       781
          7       1.00      1.00      1.00      3080

avg / total       1.00      1.00      1.00     26890

self.biases :  array([[ 0.07917589, -0.10670131,  0.03139904, -0.09398531, -0.11721164, 0.07470309, -0.05307611, -0.02180175]], dtype=float32)
Epoch number and batch_no:  24 7
Loss before optimizing :  0.816213
Loss, accuracy and verification results :  0.816212 0.704458
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.71      0.59      6086
          1       0.44      0.03      0.05       320
          2       0.76      0.86      0.81      6940
          3       0.46      0.28      0.35      1312
          4       0.00      0.00      0.00        10
          5       0.88      0.89      0.89     10336
          6       0.47      0.13      0.20      2569
          7       0.60      0.49      0.54      3695

avg / total       0.69      0.70      0.68     31268

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8437
          1       1.00      1.00      1.00        18
          2       1.00      1.00      1.00      7787
          3       1.00      1.00      1.00       803
          5       1.00      1.00      1.00     10489
          6       1.00      1.00      1.00       687
          7       1.00      1.00      1.00      3047

avg / total       1.00      1.00      1.00     31268

self.biases :  array([[ 0.07812377, -0.10741232,  0.03118096, -0.093697  , -0.11690104, 0.07519097, -0.05262629, -0.02145646]], dtype=float32)
Epoch number and batch_no:  24 8
Loss before optimizing :  0.794212
Loss, accuracy and verification results :  0.794212 0.708057
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.56      0.58      5350
          1       0.33      0.01      0.03       288
          2       0.77      0.86      0.81      6374
          3       0.41      0.27      0.32      1086
          5       0.85      0.93      0.89      9033
          6       0.43      0.26      0.32      2414
          7       0.52      0.56      0.54      3070

avg / total       0.68      0.71      0.69     27615

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4977
          1       1.00      1.00      1.00        12
          2       1.00      1.00      1.00      7161
          3       1.00      1.00      1.00       715
          5       1.00      1.00      1.00      9943
          6       1.00      1.00      1.00      1454
          7       1.00      1.00      1.00      3353

avg / total       1.00      1.00      1.00     27615

self.biases :  array([[ 0.07869749, -0.10840046,  0.03092775, -0.0944242 , -0.11663551, 0.07530384, -0.05229875, -0.0216664 ]], dtype=float32)
Epoch number and batch_no:  24 9
Loss before optimizing :  0.780608
Loss, accuracy and verification results :  0.780608 0.715009
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.63      0.60      5173
          1       1.00      0.01      0.03       281
          2       0.80      0.83      0.81      5521
          3       0.56      0.22      0.32      1037
          4       0.00      0.00      0.00        10
          5       0.85      0.94      0.89      9033
          6       0.41      0.29      0.34      2213
          7       0.57      0.52      0.54      3017

avg / total       0.70      0.72      0.70     26285

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5737
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00      5783
          3       1.00      1.00      1.00       408
          5       1.00      1.00      1.00     10027
          6       1.00      1.00      1.00      1566
          7       1.00      1.00      1.00      2760

avg / total       1.00      1.00      1.00     26285

self.biases :  array([[ 0.07949063, -0.1085063 ,  0.03100576, -0.09473288, -0.11630622, 0.07494033, -0.05277193, -0.02155745]], dtype=float32)
Epoch number and batch_no:  24 10
Loss before optimizing :  0.802024
Loss, accuracy and verification results :  0.802024 0.706087
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.76      0.60      5374
          1       0.67      0.01      0.01       299
          2       0.80      0.81      0.81      5686
          3       0.55      0.17      0.26      1150
          4       0.00      0.00      0.00        10
          5       0.89      0.90      0.89      9945
          6       0.49      0.14      0.22      2296
          7       0.55      0.49      0.52      3187

avg / total       0.71      0.71      0.69     27947

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8262
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00      5759
          3       1.00      1.00      1.00       354
          5       1.00      1.00      1.00     10020
          6       1.00      1.00      1.00       674
          7       1.00      1.00      1.00      2875

avg / total       1.00      1.00      1.00     27947

self.biases :  array([[ 0.07770943, -0.10785821,  0.03126609, -0.09405184, -0.11593512, 0.07559866, -0.05292526, -0.02129463]], dtype=float32)
Epoch number and batch_no:  24 11
Loss before optimizing :  0.774335
Loss, accuracy and verification results :  0.774335 0.718985
F1 score results : 
              precision    recall  f1-score   support

          0       0.65      0.51      0.57      4903
          1       0.80      0.08      0.14       309
          2       0.73      0.89      0.80      5644
          3       0.50      0.25      0.33       846
          4       0.00      0.00      0.00        11
          5       0.85      0.94      0.90      8751
          6       0.44      0.24      0.32      2004
          7       0.52      0.60      0.56      2947

avg / total       0.70      0.72      0.70     25415

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      3874
          1       1.00      1.00      1.00        30
          2       1.00      1.00      1.00      6914
          3       1.00      1.00      1.00       419
          5       1.00      1.00      1.00      9695
          6       1.00      1.00      1.00      1107
          7       1.00      1.00      1.00      3376

avg / total       1.00      1.00      1.00     25415

self.biases :  array([[ 0.07840946, -0.10706882,  0.03070186, -0.09411279, -0.1155308 , 0.07575607, -0.05311453, -0.02166159]], dtype=float32)
Epoch number and batch_no:  24 12
Loss before optimizing :  0.75753
Loss, accuracy and verification results :  0.75753 0.723952
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.64      0.61      5077
          1       0.32      0.04      0.08       259
          2       0.83      0.82      0.83      5902
          3       0.48      0.22      0.30      1058
          4       0.00      0.00      0.00         5
          5       0.81      0.96      0.88      9917
          6       0.51      0.18      0.26      2279
          7       0.54      0.53      0.53      2933

avg / total       0.70      0.72      0.70     27430

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5592
          1       1.00      1.00      1.00        34
          2       1.00      1.00      1.00      5829
          3       1.00      1.00      1.00       496
          5       1.00      1.00      1.00     11816
          6       1.00      1.00      1.00       809
          7       1.00      1.00      1.00      2854

avg / total       1.00      1.00      1.00     27430

self.biases :  array([[ 0.07962331, -0.10749419,  0.03106739, -0.09436602, -0.11521721, 0.07438142, -0.05238233, -0.02203336]], dtype=float32)
Epoch number and batch_no:  24 13
Loss before optimizing :  0.821907
Loss, accuracy and verification results :  0.821907 0.700322
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.77      0.59      5525
          1       0.40      0.01      0.03       272
          2       0.82      0.80      0.81      6447
          3       0.53      0.30      0.38      1256
          4       0.00      0.00      0.00         6
          5       0.90      0.88      0.89     10476
          6       0.41      0.15      0.22      2480
          7       0.57      0.44      0.50      3360

avg / total       0.71      0.70      0.69     29822

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      9050
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00      6298
          3       1.00      1.00      1.00       711
          5       1.00      1.00      1.00     10199
          6       1.00      1.00      1.00       930
          7       1.00      1.00      1.00      2624

avg / total       1.00      1.00      1.00     29822

self.biases :  array([[ 0.07794996, -0.10847027,  0.03202765, -0.09473058, -0.11503404, 0.07459877, -0.05189231, -0.02175912]], dtype=float32)
Epoch number and batch_no:  24 14
Loss before optimizing :  0.795177
Loss, accuracy and verification results :  0.795177 0.707497
F1 score results : 
              precision    recall  f1-score   support

          0       0.67      0.50      0.57      5665
          1       0.67      0.01      0.02       260
          2       0.69      0.92      0.79      6428
          3       0.48      0.21      0.29      1135
          4       0.00      0.00      0.00         5
          5       0.91      0.90      0.90      9449
          6       0.36      0.40      0.38      2417
          7       0.55      0.55      0.55      3345

avg / total       0.70      0.71      0.70     28704

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4260
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00      8522
          3       1.00      1.00      1.00       497
          5       1.00      1.00      1.00      9399
          6       1.00      1.00      1.00      2699
          7       1.00      1.00      1.00      3324

avg / total       1.00      1.00      1.00     28704

self.biases :  array([[ 0.07859263, -0.10890207,  0.03104443, -0.09469312, -0.11497201, 0.07579909, -0.05334471, -0.0215267 ]], dtype=float32)
Epoch number and batch_no:  24 15
Loss before optimizing :  0.791629
Loss, accuracy and verification results :  0.791629 0.709597
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.62      0.58      5172
          1       0.67      0.01      0.02       249
          2       0.79      0.86      0.82      6230
          3       0.54      0.17      0.26      1123
          4       0.00      0.00      0.00         5
          5       0.82      0.95      0.88      8907
          6       0.50      0.12      0.19      2294
          7       0.53      0.54      0.54      3165

avg / total       0.69      0.71      0.68     27145

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5798
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00      6830
          3       1.00      1.00      1.00       356
          5       1.00      1.00      1.00     10399
          6       1.00      1.00      1.00       542
          7       1.00      1.00      1.00      3217

avg / total       1.00      1.00      1.00     27145

self.biases :  array([[ 0.07946321, -0.10828481,  0.03009636, -0.09392726, -0.11498001, 0.07562473, -0.05350209, -0.0213807 ]], dtype=float32)
Epoch number and batch_no:  24 16
Loss before optimizing :  0.812789
Loss, accuracy and verification results :  0.812789 0.701827
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.76      0.60      5310
          1       0.67      0.01      0.02       249
          2       0.89      0.73      0.80      6257
          3       0.42      0.21      0.28      1045
          4       0.00      0.00      0.00         5
          5       0.84      0.93      0.88      9709
          6       0.61      0.06      0.12      2377
          7       0.52      0.54      0.53      3186

avg / total       0.71      0.70      0.68     28138

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8195
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00      5147
          3       1.00      1.00      1.00       521
          5       1.00      1.00      1.00     10707
          6       1.00      1.00      1.00       250
          7       1.00      1.00      1.00      3315

avg / total       1.00      1.00      1.00     28138

self.biases :  array([[ 0.07796312, -0.10723377,  0.0312274 , -0.09406365, -0.11504693, 0.07503988, -0.05208235, -0.02183082]], dtype=float32)
Epoch number and batch_no:  24 17
Loss before optimizing :  0.782728
Loss, accuracy and verification results :  0.782728 0.71783
F1 score results : 
              precision    recall  f1-score   support

          0       0.61      0.54      0.57      5290
          1       0.58      0.04      0.07       276
          2       0.77      0.87      0.82      6800
          3       0.46      0.34      0.39      1019
          4       0.00      0.00      0.00        10
          5       0.90      0.91      0.90      9057
          6       0.36      0.34      0.35      2222
          7       0.54      0.58      0.56      3178

avg / total       0.71      0.72      0.71     27852

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4724
          1       1.00      1.00      1.00        19
          2       1.00      1.00      1.00      7644
          3       1.00      1.00      1.00       758
          5       1.00      1.00      1.00      9187
          6       1.00      1.00      1.00      2096
          7       1.00      1.00      1.00      3424

avg / total       1.00      1.00      1.00     27852

self.biases :  array([[ 0.07803096, -0.10676271,  0.03213567, -0.09538028, -0.11510528, 0.07525212, -0.0524668 , -0.0225813 ]], dtype=float32)
Epoch number and batch_no:  24 18
Loss before optimizing :  0.804841
Loss, accuracy and verification results :  0.804841 0.705551
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.52      0.55      5315
          1       0.41      0.03      0.05       278
          2       0.66      0.93      0.77      5854
          3       0.53      0.20      0.29      1026
          4       0.00      0.00      0.00        25
          5       0.89      0.91      0.90      9728
          6       0.39      0.31      0.35      2326
          7       0.60      0.49      0.54      3154

avg / total       0.69      0.71      0.69     27706

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4723
          1       1.00      1.00      1.00        17
          2       1.00      1.00      1.00      8233
          3       1.00      1.00      1.00       380
          5       1.00      1.00      1.00      9891
          6       1.00      1.00      1.00      1867
          7       1.00      1.00      1.00      2595

avg / total       1.00      1.00      1.00     27706

self.biases :  array([[ 0.0795958 , -0.10729884,  0.030536  , -0.09589738, -0.114986  , 0.07629535, -0.0536414 , -0.02224486]], dtype=float32)
Epoch number and batch_no:  24 19
Loss before optimizing :  0.811752
Loss, accuracy and verification results :  0.811752 0.708277
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.77      0.60      5664
          1       0.00      0.00      0.00       305
          2       0.84      0.79      0.81      6113
          3       0.67      0.08      0.15      1204
          5       0.83      0.94      0.88     10989
          6       0.50      0.04      0.08      2377
          7       0.60      0.45      0.52      3445

avg / total       0.70      0.71      0.67     30097

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8982
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00      5748
          3       1.00      1.00      1.00       153
          5       1.00      1.00      1.00     12417
          6       1.00      1.00      1.00       209
          7       1.00      1.00      1.00      2584

avg / total       1.00      1.00      1.00     30097

self.biases :  array([[ 0.07819544, -0.10737482,  0.02989551, -0.0945313 , -0.11492437, 0.07639644, -0.05323956, -0.02082648]], dtype=float32)
Epoch number and batch_no:  24 20
Loss before optimizing :  0.834973
Loss, accuracy and verification results :  0.834973 0.697361
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.61      0.60      5226
          1       0.38      0.02      0.04       298
          2       0.84      0.78      0.81      5706
          3       0.48      0.19      0.27      1075
          5       0.82      0.94      0.87      9088
          6       0.49      0.12      0.20      2246
          7       0.43      0.65      0.52      3000

avg / total       0.69      0.70      0.67     26639

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5381
          1       1.00      1.00      1.00        16
          2       1.00      1.00      1.00      5258
          3       1.00      1.00      1.00       432
          5       1.00      1.00      1.00     10465
          6       1.00      1.00      1.00       572
          7       1.00      1.00      1.00      4515

avg / total       1.00      1.00      1.00     26639

self.biases :  array([[ 0.07774928, -0.10726624,  0.03078964, -0.09345114, -0.11492873, 0.07553801, -0.05197434, -0.02220735]], dtype=float32)
Epoch number and batch_no:  24 21
Loss before optimizing :  0.825715
Loss, accuracy and verification results :  0.825715 0.698127
F1 score results : 
              precision    recall  f1-score   support

          0       0.61      0.51      0.55      5550
          1       0.53      0.03      0.06       299
          2       0.79      0.84      0.81      6710
          3       0.32      0.37      0.34      1182
          4       0.00      0.00      0.00        11
          5       0.86      0.91      0.89      9559
          6       0.37      0.39      0.38      2467
          7       0.55      0.51      0.53      3264

avg / total       0.69      0.70      0.69     29042

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4664
          1       1.00      1.00      1.00        17
          2       1.00      1.00      1.00      7189
          3       1.00      1.00      1.00      1365
          5       1.00      1.00      1.00     10142
          6       1.00      1.00      1.00      2647
          7       1.00      1.00      1.00      3018

avg / total       1.00      1.00      1.00     29042

self.biases :  array([[ 0.07901042, -0.10772236,  0.03209126, -0.09536824, -0.11490013, 0.0752131 , -0.05265483, -0.02332647]], dtype=float32)
Epoch number and batch_no:  24 22
Loss before optimizing :  0.794484
Loss, accuracy and verification results :  0.794484 0.71045
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.69      0.59      5412
          1       0.00      0.00      0.00       279
          2       0.72      0.88      0.79      6011
          3       0.47      0.15      0.23      1099
          4       0.00      0.00      0.00        17
          5       0.89      0.91      0.90     10491
          6       0.41      0.22      0.29      2319
          7       0.66      0.38      0.48      3234

avg / total       0.69      0.71      0.69     28862

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      7264
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00      7310
          3       1.00      1.00      1.00       354
          5       1.00      1.00      1.00     10802
          6       1.00      1.00      1.00      1244
          7       1.00      1.00      1.00      1887

avg / total       1.00      1.00      1.00     28862

self.biases :  array([[ 0.07890058, -0.10780241,  0.03201717, -0.09646016, -0.11477714, 0.07571667, -0.05375372, -0.02208827]], dtype=float32)
Epoch number and batch_no:  24 23
Loss before optimizing :  0.840098
Loss, accuracy and verification results :  0.840098 0.7006
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.65      0.59      5483
          1       0.73      0.05      0.09       335
          2       0.71      0.88      0.79      5570
          3       0.73      0.12      0.20      1077
          5       0.83      0.94      0.88      9207
          6       0.56      0.09      0.16      2365
          7       0.58      0.50      0.54      3281

avg / total       0.69      0.70      0.66     27318

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6591
          1       1.00      1.00      1.00        22
          2       1.00      1.00      1.00      6897
          3       1.00      1.00      1.00       173
          5       1.00      1.00      1.00     10436
          6       1.00      1.00      1.00       388
          7       1.00      1.00      1.00      2811

avg / total       1.00      1.00      1.00     27318

self.biases :  array([[ 0.07849568, -0.10707024,  0.03071881, -0.09551328, -0.11476803, 0.07536051, -0.05329784, -0.02019043]], dtype=float32)
Epoch number and batch_no:  24 24
Loss before optimizing :  0.820206
Loss, accuracy and verification results :  0.820206 0.700158
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.61      0.60      4672
          1       0.33      0.01      0.02       228
          2       0.85      0.78      0.81      5604
          3       0.60      0.11      0.18       939
          5       0.83      0.91      0.87      8085
          6       0.49      0.12      0.19      1916
          7       0.41      0.72      0.53      2602

avg / total       0.70      0.70      0.68     24046

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4884
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00      5140
          3       1.00      1.00      1.00       171
          5       1.00      1.00      1.00      8850
          6       1.00      1.00      1.00       448
          7       1.00      1.00      1.00      4547

avg / total       1.00      1.00      1.00     24046

self.biases :  array([[ 0.07874524, -0.10691094,  0.0312517 , -0.09353626, -0.1148795 , 0.07466471, -0.05220991, -0.02255987]], dtype=float32)
Epoch number and batch_no:  24 25
Loss before optimizing :  0.7919
Loss, accuracy and verification results :  0.7919 0.713835
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.68      0.60      4901
          1       0.30      0.01      0.02       254
          2       0.82      0.82      0.82      5423
          3       0.37      0.41      0.39      1105
          4       0.00      0.00      0.00         5
          5       0.90      0.90      0.90      9332
          6       0.38      0.30      0.33      2047
          7       0.60      0.46      0.52      2932

avg / total       0.71      0.71      0.71     25999

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6210
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00      5418
          3       1.00      1.00      1.00      1231
          5       1.00      1.00      1.00      9317
          6       1.00      1.00      1.00      1582
          7       1.00      1.00      1.00      2231

avg / total       1.00      1.00      1.00     25999

self.biases :  array([[ 0.07851457, -0.10772689,  0.03243906, -0.09427661, -0.11505549, 0.07520741, -0.05264975, -0.0238317 ]], dtype=float32)
Epoch number and batch_no:  24 26
Loss before optimizing :  0.807878
Loss, accuracy and verification results :  0.807878 0.708842
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.63      0.58      5138
          1       1.00      0.01      0.02       264
          2       0.71      0.89      0.79      5673
          3       0.41      0.33      0.36       997
          4       0.00      0.00      0.00        10
          5       0.88      0.91      0.90      9868
          6       0.39      0.28      0.33      2245
          7       0.72      0.36      0.48      3151

avg / total       0.70      0.71      0.69     27346

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5997
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00      7077
          3       1.00      1.00      1.00       798
          5       1.00      1.00      1.00     10272
          6       1.00      1.00      1.00      1629
          7       1.00      1.00      1.00      1571

avg / total       1.00      1.00      1.00     27346

self.biases :  array([[ 0.07816944, -0.10825361,  0.03209044, -0.09616827, -0.11516436, 0.07631943, -0.0538895 , -0.02207556]], dtype=float32)
Epoch number and batch_no:  24 27
Loss before optimizing :  0.835478
Loss, accuracy and verification results :  0.835478 0.699123
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.60      0.59      5215
          1       0.40      0.01      0.01       324
          2       0.72      0.89      0.80      5658
          3       0.60      0.10      0.17      1141
          4       0.00      0.00      0.00         5
          5       0.81      0.95      0.88      9497
          6       0.53      0.13      0.20      2475
          7       0.52      0.50      0.51      3271

avg / total       0.67      0.70      0.66     27586

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5567
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00      6977
          3       1.00      1.00      1.00       190
          5       1.00      1.00      1.00     11108
          6       1.00      1.00      1.00       584
          7       1.00      1.00      1.00      3155

avg / total       1.00      1.00      1.00     27586

self.biases :  array([[ 0.07847539, -0.10756827,  0.0304428 , -0.09609248, -0.1152446 , 0.07604292, -0.05351998, -0.0200578 ]], dtype=float32)
Epoch number and batch_no:  24 28
Loss before optimizing :  0.797653
Loss, accuracy and verification results :  0.797653 0.705795
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.64      0.61      5217
          1       0.57      0.01      0.03       275
          2       0.85      0.78      0.81      5453
          3       0.66      0.09      0.15      1097
          5       0.87      0.92      0.90      9710
          6       0.54      0.10      0.17      2121
          7       0.40      0.72      0.52      3013

avg / total       0.72      0.71      0.68     26886

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5787
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00      4968
          3       1.00      1.00      1.00       145
          5       1.00      1.00      1.00     10208
          6       1.00      1.00      1.00       407
          7       1.00      1.00      1.00      5364

avg / total       1.00      1.00      1.00     26886

self.biases :  array([[ 0.0789755 , -0.1070452 ,  0.03019514, -0.09425484, -0.11535253, 0.07584233, -0.05235997, -0.0224213 ]], dtype=float32)
Epoch number and batch_no:  24 29
Loss before optimizing :  0.811517
Loss, accuracy and verification results :  0.811517 0.703763
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.74      0.60      4815
          1       0.45      0.05      0.08       305
          2       0.87      0.75      0.80      5493
          3       0.41      0.28      0.33       987
          5       0.88      0.93      0.90      8631
          6       0.38      0.26      0.31      2112
          7       0.59      0.42      0.49      2799

avg / total       0.71      0.70      0.69     25142

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      7115
          1       1.00      1.00      1.00        31
          2       1.00      1.00      1.00      4728
          3       1.00      1.00      1.00       677
          5       1.00      1.00      1.00      9141
          6       1.00      1.00      1.00      1462
          7       1.00      1.00      1.00      1988

avg / total       1.00      1.00      1.00     25142

self.biases :  array([[ 0.07774952, -0.10751382,  0.03174043, -0.09362643, -0.11548319, 0.0759009 , -0.05260671, -0.02329716]], dtype=float32)
Epoch number and batch_no:  24 30
Loss before optimizing :  0.78359
Loss, accuracy and verification results :  0.78359 0.710719
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.52      0.57      5207
          1       0.46      0.03      0.05       237
          2       0.73      0.90      0.81      6258
          3       0.32      0.41      0.36       989
          5       0.86      0.94      0.90      9205
          6       0.36      0.38      0.37      2293
          7       0.69      0.35      0.46      3006

avg / total       0.70      0.71      0.70     27195

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4325
          1       1.00      1.00      1.00        13
          2       1.00      1.00      1.00      7633
          3       1.00      1.00      1.00      1278
          5       1.00      1.00      1.00     10049
          6       1.00      1.00      1.00      2383
          7       1.00      1.00      1.00      1514

avg / total       1.00      1.00      1.00     27195

self.biases :  array([[ 0.07852234, -0.10856073,  0.03227547, -0.09569729, -0.11564005, 0.0756449 , -0.05407729, -0.02161125]], dtype=float32)
Epoch number and batch_no:  24 31
Loss before optimizing :  0.769116
Loss, accuracy and verification results :  0.769116 0.722711
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.60      0.59      4748
          1       0.00      0.00      0.00       279
          2       0.71      0.92      0.80      5401
          3       0.56      0.19      0.28       970
          5       0.86      0.94      0.90      8784
          6       0.54      0.13      0.20      1964
          7       0.58      0.54      0.56      2864

avg / total       0.70      0.72      0.69     25010

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4893
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00      7077
          3       1.00      1.00      1.00       331
          5       1.00      1.00      1.00      9590
          6       1.00      1.00      1.00       461
          7       1.00      1.00      1.00      2656

avg / total       1.00      1.00      1.00     25010

self.biases :  array([[ 0.07995351, -0.10840041,  0.03101284, -0.0967247 , -0.11581582, 0.07507031, -0.05432675, -0.01969656]], dtype=float32)
Epoch number and batch_no:  24 32
Loss before optimizing :  0.846976
Loss, accuracy and verification results :  0.846976 0.692065
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.71      0.60      5881
          1       0.00      0.00      0.00       339
          2       0.82      0.80      0.81      7119
          3       0.69      0.07      0.13      1262
          5       0.88      0.89      0.89      9293
          6       0.69      0.04      0.08      2480
          7       0.44      0.66      0.52      3392

avg / total       0.71      0.69      0.66     29766

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8007
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00      6965
          3       1.00      1.00      1.00       131
          5       1.00      1.00      1.00      9382
          6       1.00      1.00      1.00       160
          7       1.00      1.00      1.00      5120

avg / total       1.00      1.00      1.00     29766

self.biases :  array([[ 0.07944612, -0.10719723,  0.03060527, -0.09542882, -0.1160223 , 0.07505858, -0.05253171, -0.02127934]], dtype=float32)
Epoch number and batch_no:  24 33
Loss before optimizing :  0.793017
Loss, accuracy and verification results :  0.793017 0.711866
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.70      0.60      4786
          1       0.54      0.02      0.05       283
          2       0.82      0.80      0.81      4777
          3       0.50      0.15      0.24      1053
          5       0.86      0.93      0.89      9143
          6       0.45      0.22      0.30      2137
          7       0.54      0.51      0.52      2858

avg / total       0.70      0.71      0.69     25037

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6332
          1       1.00      1.00      1.00        13
          2       1.00      1.00      1.00      4699
          3       1.00      1.00      1.00       324
          5       1.00      1.00      1.00      9953
          6       1.00      1.00      1.00      1048
          7       1.00      1.00      1.00      2668

avg / total       1.00      1.00      1.00     25037

self.biases :  array([[ 0.07826986, -0.10644323,  0.03096126, -0.09349435, -0.11626986, 0.07488962, -0.05131477, -0.02260619]], dtype=float32)
Epoch number and batch_no:  24 34
Loss before optimizing :  0.817442
Loss, accuracy and verification results :  0.817442 0.693904
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.46      0.54      5572
          1       0.24      0.06      0.10       277
          2       0.79      0.85      0.82      6112
          3       0.32      0.43      0.37      1158
          4       0.00      0.00      0.00         5
          5       0.87      0.92      0.89      9929
          6       0.31      0.56      0.40      2418
          7       0.68      0.35      0.46      3154

avg / total       0.71      0.69      0.69     28625

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      3898
          1       1.00      1.00      1.00        71
          2       1.00      1.00      1.00      6599
          3       1.00      1.00      1.00      1558
          5       1.00      1.00      1.00     10547
          6       1.00      1.00      1.00      4351
          7       1.00      1.00      1.00      1601

avg / total       1.00      1.00      1.00     28625

self.biases :  array([[ 0.07959015, -0.10813498,  0.03160815, -0.09469204, -0.11651908, 0.07488431, -0.05363546, -0.02169225]], dtype=float32)
Epoch number and batch_no:  24 35
Loss before optimizing :  0.75907
Loss, accuracy and verification results :  0.75907 0.726096
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.70      0.61      4761
          1       0.25      0.00      0.01       283
          2       0.78      0.87      0.82      5152
          3       0.49      0.29      0.37      1116
          5       0.87      0.93      0.90      9885
          6       0.54      0.13      0.21      2069
          7       0.57      0.47      0.51      2776

avg / total       0.71      0.73      0.70     26042

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6313
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00      5751
          3       1.00      1.00      1.00       662
          5       1.00      1.00      1.00     10542
          6       1.00      1.00      1.00       490
          7       1.00      1.00      1.00      2280

avg / total       1.00      1.00      1.00     26042

self.biases :  array([[ 0.07989886, -0.10933239,  0.0317219 , -0.09601326, -0.11682158, 0.07507545, -0.05472383, -0.02027796]], dtype=float32)
Epoch number and batch_no:  24 36
Loss before optimizing :  0.808002
Loss, accuracy and verification results :  0.808002 0.711731
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.69      0.61      5607
          1       0.00      0.00      0.00       302
          2       0.76      0.84      0.80      5834
          3       0.62      0.10      0.17      1164
          5       0.88      0.91      0.90     10634
          6       0.69      0.03      0.06      2327
          7       0.48      0.62      0.54      3150

avg / total       0.71      0.71      0.68     29018

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      7159
          2       1.00      1.00      1.00      6468
          3       1.00      1.00      1.00       191
          5       1.00      1.00      1.00     11026
          6       1.00      1.00      1.00       110
          7       1.00      1.00      1.00      4064

avg / total       1.00      1.00      1.00     29018

self.biases :  array([[ 0.07908238, -0.10937514,  0.03131425, -0.09569722, -0.11715259, 0.0756515 , -0.05348292, -0.02091847]], dtype=float32)
Epoch number and batch_no:  24 37
Loss before optimizing :  0.816881
Loss, accuracy and verification results :  0.816881 0.709143
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.62      0.60      5651
          1       0.00      0.00      0.00       301
          2       0.76      0.87      0.81      6644
          3       0.70      0.16      0.26      1209
          4       0.00      0.00      0.00         5
          5       0.84      0.94      0.89      8918
          6       0.47      0.15      0.23      2425
          7       0.51      0.58      0.55      3294

avg / total       0.69      0.71      0.68     28447

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6023
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00      7559
          3       1.00      1.00      1.00       278
          5       1.00      1.00      1.00     10042
          6       1.00      1.00      1.00       792
          7       1.00      1.00      1.00      3752

avg / total       1.00      1.00      1.00     28447

self.biases :  array([[ 0.07887617, -0.1083409 ,  0.03048332, -0.09386648, -0.11742368, 0.07540322, -0.05169147, -0.02216252]], dtype=float32)
Epoch number and batch_no:  24 38
Loss before optimizing :  0.781797
Loss, accuracy and verification results :  0.781797 0.713087
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.55      0.57      5004
          1       0.20      0.00      0.01       256
          2       0.80      0.83      0.82      6395
          3       0.42      0.30      0.35       899
          5       0.87      0.94      0.90      9120
          6       0.32      0.44      0.37      2153
          7       0.62      0.40      0.49      2871

avg / total       0.71      0.71      0.71     26698

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      4646
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00      6644
          3       1.00      1.00      1.00       634
          5       1.00      1.00      1.00      9890
          6       1.00      1.00      1.00      3022
          7       1.00      1.00      1.00      1857

avg / total       1.00      1.00      1.00     26698

self.biases :  array([[ 0.07955339, -0.1074403 ,  0.03046544, -0.0933149 , -0.11768756, 0.07512727, -0.05287499, -0.02187489]], dtype=float32)
Epoch number and batch_no:  24 39
Loss before optimizing :  0.762326
Loss, accuracy and verification results :  0.762326 0.721941
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.70      0.61      5003
          1       0.47      0.03      0.05       274
          2       0.81      0.82      0.81      6194
          3       0.38      0.39      0.38       948
          5       0.90      0.92      0.91      8795
          6       0.42      0.22      0.29      2118
          7       0.61      0.50      0.55      2943

avg / total       0.72      0.72      0.71     26275

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6578
          1       1.00      1.00      1.00        17
          2       1.00      1.00      1.00      6238
          3       1.00      1.00      1.00       976
          5       1.00      1.00      1.00      9004
          6       1.00      1.00      1.00      1075
          7       1.00      1.00      1.00      2387

avg / total       1.00      1.00      1.00     26275

self.biases :  array([[ 0.0793349 , -0.10821328,  0.03122859, -0.09509332, -0.11794056, 0.07575665, -0.05436617, -0.02088144]], dtype=float32)
Epoch number and batch_no:  24 40
Loss before optimizing :  0.782489
Loss, accuracy and verification results :  0.782489 0.717525
F1 score results : 
              precision    recall  f1-score   support

          0       0.56      0.64      0.59      4890
          1       0.00      0.00      0.00       286
          2       0.77      0.86      0.81      5426
          3       0.49      0.12      0.19       989
          5       0.85      0.93      0.89      9886
          6       0.62      0.08      0.14      2030
          7       0.50      0.58      0.54      2998

avg / total       0.70      0.72      0.69     26505

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5580
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00      6062
          3       1.00      1.00      1.00       241
          5       1.00      1.00      1.00     10870
          6       1.00      1.00      1.00       249
          7       1.00      1.00      1.00      3499

avg / total       1.00      1.00      1.00     26505

self.biases :  array([[ 0.07930995, -0.10898519,  0.0316079 , -0.09605408, -0.11818439, 0.07590245, -0.05426519, -0.02090688]], dtype=float32)
Epoch number and batch_no:  24 41
Loss before optimizing :  0.777578
Loss, accuracy and verification results :  0.777578 0.717204
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.60      0.59      4801
          1       0.56      0.05      0.09       293
          2       0.77      0.89      0.82      6319
          3       0.71      0.09      0.16      1102
          5       0.83      0.96      0.89      8454
          6       0.58      0.09      0.16      2053
          7       0.50      0.58      0.53      2827

avg / total       0.70      0.72      0.68     25849

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5040
          1       1.00      1.00      1.00        25
          2       1.00      1.00      1.00      7302
          3       1.00      1.00      1.00       140
          5       1.00      1.00      1.00      9749
          6       1.00      1.00      1.00       321
          7       1.00      1.00      1.00      3272

avg / total       1.00      1.00      1.00     25849

self.biases :  array([[ 0.07973876, -0.10890932,  0.03129011, -0.09489103, -0.11841858, 0.0750198 , -0.05265643, -0.02189067]], dtype=float32)
Epoch number and batch_no:  24 42
Loss before optimizing :  0.761563
Loss, accuracy and verification results :  0.761564 0.723168
F1 score results : 
              precision    recall  f1-score   support

          0       0.56      0.66      0.61      5476
          1       1.00      0.01      0.02       278
          2       0.77      0.88      0.82      6276
          3       0.47      0.19      0.27       934
          4       0.00      0.00      0.00         5
          5       0.88      0.93      0.90      9168
          6       0.46      0.24      0.31      2468
          7       0.58      0.52      0.55      3011

avg / total       0.71      0.72      0.70     27616

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      6423
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00      7195
          3       1.00      1.00      1.00       373
          5       1.00      1.00      1.00      9644
          6       1.00      1.00      1.00      1274
          7       1.00      1.00      1.00      2704

avg / total       1.00      1.00      1.00     27616

self.biases :  array([[ 0.08008382, -0.10860634,  0.03051052, -0.09386805, -0.1185897 , 0.07446101, -0.05127084, -0.02235139]], dtype=float32)
Epoch number and testing batch number :  24 0
Loss, accuracy and verification results :  0.935416 0.662224
F1 score results : 
              precision    recall  f1-score   support

          0       0.22      0.40      0.28      4163
          1       0.00      0.00      0.00       277
          2       0.22      0.22      0.22      4224
          3       0.02      0.01      0.01       636
          4       0.00      0.00      0.00         5
          5       0.30      0.27      0.29      6345
          6       0.09      0.03      0.05      1859
          7       0.12      0.08      0.09      2374

avg / total       0.21      0.23      0.21     19883

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      7597
          2       1.00      1.00      1.00      4170
          3       1.00      1.00      1.00       224
          5       1.00      1.00      1.00      5728
          6       1.00      1.00      1.00       706
          7       1.00      1.00      1.00      1458

avg / total       1.00      1.00      1.00     19883

Epoch number and testing batch number :  24 1
Loss, accuracy and verification results :  0.939677 0.662754
F1 score results : 
              precision    recall  f1-score   support

          0       0.22      0.38      0.28      4030
          1       0.00      0.00      0.00       270
          2       0.22      0.23      0.23      4228
          3       0.03      0.01      0.01       809
          4       0.00      0.00      0.00         5
          5       0.31      0.30      0.30      6259
          6       0.11      0.04      0.06      1895
          7       0.12      0.08      0.10      2353

avg / total       0.22      0.23      0.22     19849

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      7038
          2       1.00      1.00      1.00      4341
          3       1.00      1.00      1.00       219
          5       1.00      1.00      1.00      5997
          6       1.00      1.00      1.00       750
          7       1.00      1.00      1.00      1504

avg / total       1.00      1.00      1.00     19849

Epoch number and testing batch number :  24 2
Loss, accuracy and verification results :  0.983412 0.640353
F1 score results : 
              precision    recall  f1-score   support

          0       0.22      0.41      0.29      4555
          1       0.00      0.00      0.00       271
          2       0.25      0.23      0.24      4765
          3       0.03      0.01      0.01       819
          4       0.00      0.00      0.00        10
          5       0.29      0.28      0.29      6123
          6       0.09      0.03      0.05      2067
          7       0.11      0.06      0.08      2558

avg / total       0.21      0.23      0.21     21168

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      8457
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00      4390
          3       1.00      1.00      1.00       213
          5       1.00      1.00      1.00      5895
          6       1.00      1.00      1.00       730
          7       1.00      1.00      1.00      1482

avg / total       1.00      1.00      1.00     21168

Epoch number and testing batch number :  24 3
Loss, accuracy and verification results :  0.99877 0.638593
F1 score results : 
              precision    recall  f1-score   support

          0       0.22      0.43      0.29      4863
          1       0.00      0.00      0.00       322
          2       0.21      0.21      0.21      4548
          3       0.03      0.01      0.01       793
          4       0.00      0.00      0.00        10
          5       0.30      0.26      0.28      7020
          6       0.10      0.03      0.05      2366
          7       0.13      0.08      0.10      2543

avg / total       0.21      0.23      0.21     22465

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00      9245
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00      4554
          3       1.00      1.00      1.00       236
          5       1.00      1.00      1.00      6132
          6       1.00      1.00      1.00       749
          7       1.00      1.00      1.00      1547

avg / total       1.00      1.00      1.00     22465




"""














