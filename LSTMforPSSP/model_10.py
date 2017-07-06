# removed all maxpooling sliding windows + relu was introduced + increased the number of hudden units to 200 + weighted gates
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
    hidden_units = 200,
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

    # to use xavier initialization, dtype needs to be float32
    self.hidden_units = tf.constant(hidden_units, dtype = tf.float32)
    
    # define weights and biases here (8 weights + 1 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_gate_1 = tf.Variable(tf.random_uniform(shape=[hidden_units * 2 + 122, hidden_units * 2], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.weight_gate_2 = tf.Variable(tf.random_uniform(shape=[hidden_units * 2 + 122, 122], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.weight_h = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units * 2 + 122, hidden_units * 2 + 122], maxval=1, dtype=tf.float32) / tf.sqrt((self.hidden_units * 2 + 122) / 2), dtype=tf.float32) 
    self.weight_y = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units * 2 + 122, num_classes], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.biases_h = tf.Variable(tf.zeros([hidden_units * 2 + 122], dtype=tf.float32), dtype=tf.float32)
    self.biases_y = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    self.biases_gate_1 = tf.Variable(tf.zeros([hidden_units * 2], dtype=tf.float32), dtype=tf.float32)
    self.biases_gate_2 = tf.Variable(tf.zeros([122], dtype=tf.float32), dtype=tf.float32)
    
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

    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ batch_size, 700, 200])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ batch_size, 700, 200])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 200])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 200])
    
    list_of_tensors = [self.outputs_f_c_r, self.outputs_b_c_r ]

    self.input_x_r = tf.reshape(self.input_x[:, 50:750, :], [-1, 122])
    self.outputs_rnn_concat = tf.concat(list_of_tensors, axis = 1)
    self.op_rnn_and_inp_concat = tf.concat([self.input_x_r, self.outputs_rnn_concat], axis = 1)

    self.output_gate_1 = tf.sigmoid(tf.matmul(self.op_rnn_and_inp_concat, self.weight_gate_1) + self.biases_gate_1)
    self.output_gate_2 = tf.sigmoid(tf.matmul(self.op_rnn_and_inp_concat, self.weight_gate_2) + self.biases_gate_2)
    self.outputs_rnn_concat_gated = tf.multiply(self.output_gate_1, self.outputs_rnn_concat)
    self.input_x_r_gated = tf.multiply(self.output_gate_2, self.input_x_r)

    self.op_rnn_and_inp_concat_gated = tf.concat([self.input_x_r_gated, self.outputs_rnn_concat_gated], axis = 1)
    self.h_predicted = tf.nn.relu(tf.matmul(self.op_rnn_and_inp_concat_gated, self.weight_h) + self.biases_h) 
    self.y_predicted = (tf.matmul(self.h_predicted, self.weight_y) + self.biases_y) 

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

  def print_biases(self, x, y, seq_len, msks):
    biases = self.sess.run([
      self.biases_y],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    print("self.biases : ", np.array_repr(np.array(biases)).replace('\n', '').replace(' ', ''))

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
    print("self.outputs_rnn_concat.shape", self.outputs_rnn_concat.shape)
    print("self.weight_gate_1.shape", self.weight_gate_1.shape)
    print("self.weight_gate_2.shape", self.weight_gate_2.shape)
  
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
  model_path = "./data/LSTMmodels/"
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
      # print("Loss before optimizing : ", loss_reduced)
      loss, accuracy, no_of_entries_unmasked = model.optimize_mini(x_inp, y_inp, l_inp, m_inp)
      print("Loss and accuracy : ", loss, accuracy)
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
      print("Loss and accuracy : ", loss, accuracy)
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
    print("Epoch number and 'current' results on train data : ", acc_train_avg, loss_train_avg)
    print("Epoch number and 'current' results on test data  : ", acc_test_avg, loss_test_avg)
    epoch_wise_accs.append([acc_train_avg, acc_test_avg])
    epoch_wise_loss.append([loss_train_avg, loss_test_avg])
    print("\n\nPrinting all previous results : \n")
    for i in range(len(epoch_wise_accs)):
      print("Epoch number, train and test accuracy  :  ", i, epoch_wise_accs[i], "\n")
      print("Epoch number, train and test loss      :  ", i,epoch_wise_loss[i], "\n")
    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##
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
Epoch no - 25

Printing all previous results : 
Printing all previous results : 

Epoch number, train and test accuracy  :   0 [0.34945834111855473, 0.42059312760829926] 

Epoch number, train and test loss      :   0 [1.7108710117118304, 1.4595310091972351] 

Epoch number, train and test accuracy  :   1 [0.53396268362222721, 0.5915553867816925] 

Epoch number, train and test loss      :   1 [1.2605035277300103, 1.135225236415863] 

Epoch number, train and test accuracy  :   2 [0.6418034406595452, 0.62684845924377441] 

Epoch number, train and test loss      :   2 [0.99861942474232168, 1.0331979990005493] 

Epoch number, train and test accuracy  :   3 [0.66696714939073076, 0.64594613015651703] 

Epoch number, train and test loss      :   3 [0.92129139428914975, 0.97832785546779633] 

Epoch number, train and test accuracy  :   4 [0.68095071787057926, 0.65528813004493713] 

Epoch number, train and test loss      :   4 [0.87988064039585201, 0.95413900911808014] 

Epoch number, train and test accuracy  :   5 [0.68861926156421038, 0.66096389293670654] 

Epoch number, train and test loss      :   5 [0.85665048693501678, 0.94021928310394287] 

Epoch number, train and test accuracy  :   6 [0.6966637705647668, 0.66628466546535492] 

Epoch number, train and test loss      :   6 [0.83572653559751287, 0.9261510968208313] 

Epoch number, train and test accuracy  :   7 [0.70347784979398864, 0.67189757525920868] 

Epoch number, train and test loss      :   7 [0.81712041621984433, 0.91263020038604736] 

Epoch number, train and test accuracy  :   8 [0.70923640284427381, 0.67558345198631287] 

Epoch number, train and test loss      :   8 [0.80057161353355233, 0.90697205066680908] 

Epoch number, train and test accuracy  :   9 [0.71460442210352693, 0.67698587477207184] 

Epoch number, train and test loss      :   9 [0.78583539363949795, 0.9036860316991806] 

Epoch number, train and test accuracy  :   10 [0.71965698447338367, 0.67717447876930237] 

Epoch number, train and test loss      :   10 [0.7716669093730838, 0.90323750674724579] 

Epoch number, train and test accuracy  :   11 [0.72387536875037262, 0.6768517941236496] 

Epoch number, train and test loss      :   11 [0.75937218860138289, 0.90587890148162842] 

Epoch number, train and test accuracy  :   12 [0.72760956509168762, 0.67578762769699097] 

Epoch number, train and test loss      :   12 [0.7484740379244782, 0.91129274666309357] 

Epoch number, train and test accuracy  :   13 [0.73065295607544656, 0.6726585328578949] 

Epoch number, train and test loss      :   13 [0.740111461905546, 0.91574414074420929] 

Epoch number, train and test accuracy  :   14 [0.73063507745432299, 0.67470580339431763] 

Epoch number, train and test loss      :   14 [0.73857985956724304, 0.91649757325649261] 

Epoch number, train and test accuracy  :   15 [0.73539271881414014, 0.66822119057178497] 

Epoch number, train and test loss      :   15 [0.72700417734855827, 0.9323686808347702] 

Epoch number, train and test accuracy  :   16 [0.73573372807613635, 0.66719335317611694] 

Epoch number, train and test loss      :   16 [0.72326149496921277, 0.93812640011310577] 

Epoch number, train and test accuracy  :   17 [0.73606520752574123, 0.67073149979114532] 

Epoch number, train and test loss      :   17 [0.72245578294576596, 0.92772842943668365] 

Epoch number, train and test accuracy  :   18 [0.73749025062073104, 0.66905668377876282] 

Epoch number, train and test loss      :   18 [0.71755988930546966, 0.94245891273021698] 

Epoch number, train and test accuracy  :   19 [0.74108941749084822, 0.66095715761184692] 

Epoch number, train and test loss      :   19 [0.70829716532729392, 0.97997061908245087] 

Epoch number, train and test accuracy  :   20 [0.74249815247779671, 0.6649308055639267] 

Epoch number, train and test loss      :   20 [0.70300117897432901, 0.968647301197052] 

Epoch number, train and test accuracy  :   21 [0.74456307361292284, 0.66855348646640778] 

Epoch number, train and test loss      :   21 [0.69772952240566877, 0.95208144187927246] 

Epoch number, train and test accuracy  :   22 [0.74905224179112639, 0.66724741458892822] 

Epoch number, train and test loss      :   22 [0.68422122334325042, 0.95971889793872833] 

Epoch number, train and test accuracy  :   23 [0.75379851806995479, 0.66412055492401123] 

Epoch number, train and test loss      :   23 [0.67022917852845298, 0.9735255092382431] 

Epoch number, train and test accuracy  :   24 [0.757236429425173, 0.66503764688968658] 

Epoch number, train and test loss      :   24 [0.65893022265545154, 0.97607101500034332]


"""









