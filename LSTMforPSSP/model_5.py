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

    # to use xavier initialization, dtype needs to be float32
    self.hidden_units = tf.constant(hidden_units, dtype = tf.float32)
    
    # define weights and biases here (8 weights + 1 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_total = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units * 2, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_gate_1 = tf.Variable(tf.random_uniform(shape=[hidden_units * 2 + 122, hidden_units * 2], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.weight_gate_2 = tf.Variable(tf.random_uniform(shape=[hidden_units * 2 + 122, 122], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.weight_h = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units * 2 + 122, num_classes], maxval=1, dtype=tf.float32) / tf.sqrt(self.hidden_units * 2 + 122), dtype=tf.float32) 
    self.biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
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

    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ batch_size, 700, 100])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ batch_size, 700, 100])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 100])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 100])
    
    list_of_tensors = [self.outputs_f_c_r, self.outputs_b_c_r ]

    self.input_x_r = tf.reshape(self.input_x[:, 50:750, :], [-1, 122])
    self.outputs_rnn_concat = tf.concat(list_of_tensors, axis = 1)
    self.op_rnn_and_inp_concat = tf.concat([self.input_x_r, self.outputs_rnn_concat], axis = 1)

    self.output_gate_1 = tf.sigmoid(tf.matmul(self.op_rnn_and_inp_concat, self.weight_gate_1) + self.biases_gate_1)
    self.output_gate_2 = tf.sigmoid(tf.matmul(self.op_rnn_and_inp_concat, self.weight_gate_2) + self.biases_gate_2)
    self.outputs_rnn_concat_gated = tf.multiply(self.output_gate_1, self.outputs_rnn_concat)
    self.input_x_r_gated = tf.multiply(self.output_gate_2, self.input_x_r)

    self.op_rnn_and_inp_concat_gated = tf.concat([self.input_x_r_gated, self.outputs_rnn_concat_gated], axis = 1)
    self.y_predicted = tf.matmul(self.op_rnn_and_inp_concat_gated, self.weight_h) + self.biases 

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
      self.biases],
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
      print("Epoch number, train and test accuracy  :  ", epoch_wise_accs[i], "\n")
      print("Epoch number, train and test loss      :  ", epoch_wise_loss[i], "\n")
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
Epoch 27 :

Printing all previous results : 

Epoch number, train and test accuracy  :   [0.45500605640023256, 0.54526132345199585] 

Epoch number, train and test loss      :   [1.4481266709261162, 1.2433373034000397] 

Epoch number, train and test accuracy  :   [0.63411329790603288, 0.61940868198871613] 

Epoch number, train and test loss      :   [1.0213782482369, 1.0500211417675018] 

Epoch number, train and test accuracy  :   [0.66622809476630629, 0.64414221048355103] 

Epoch number, train and test loss      :   [0.9248536783595418, 0.98089505732059479] 

Epoch number, train and test accuracy  :   [0.68124265310376186, 0.65520527958869934] 

Epoch number, train and test loss      :   [0.88080192721167272, 0.95140770077705383] 

Epoch number, train and test accuracy  :   [0.69078928648039351, 0.66251355409622192] 

Epoch number, train and test loss      :   [0.85393707558166154, 0.93363100290298462] 

Epoch number, train and test accuracy  :   [0.6979942668315976, 0.66654200851917267] 

Epoch number, train and test loss      :   [0.83454903752304788, 0.92268088459968567] 

Epoch number, train and test accuracy  :   [0.70329102943109911, 0.67018109560012817] 

Epoch number, train and test loss      :   [0.81952846050262451, 0.91629363596439362] 

Epoch number, train and test accuracy  :   [0.70753479974214417, 0.67076365649700165] 

Epoch number, train and test loss      :   [0.80718129596044852, 0.91658078134059906] 

Epoch number, train and test accuracy  :   [0.71125102043151855, 0.67334127426147461] 

Epoch number, train and test loss      :   [0.79659260012382682, 0.90937651693820953] 

Epoch number, train and test accuracy  :   [0.71268010832542594, 0.67001509666442871] 

Epoch number, train and test loss      :   [0.79164338804954704, 0.91508010029792786] 

Epoch number, train and test accuracy  :   [0.71538433363271314, 0.67344294488430023] 

Epoch number, train and test loss      :   [0.7847482922465302, 0.90353420376777649] 

Epoch number, train and test accuracy  :   [0.71775166517080258, 0.67423772811889648] 

Epoch number, train and test loss      :   [0.7783324510552162, 0.90421183407306671] 

Epoch number, train and test accuracy  :   [0.72180635707322938, 0.67255474627017975] 

Epoch number, train and test loss      :   [0.76585313885710959, 0.91124261915683746] 

Epoch number, train and test accuracy  :   [0.72440207836239834, 0.67372137308120728] 

Epoch number, train and test loss      :   [0.75793137661246368, 0.91061103343963623] 

Epoch number, train and test accuracy  :   [0.72666328868200614, 0.6724407821893692] 

Epoch number, train and test loss      :   [0.7519576591114665, 0.91777980327606201] 

Epoch number, train and test accuracy  :   [0.72728001932765163, 0.66926649212837219] 

Epoch number, train and test loss      :   [0.74868468212526895, 0.93661564588546753] 

Epoch number, train and test accuracy  :   [0.7253233513166738, 0.66958259046077728] 

Epoch number, train and test loss      :   [0.7546223928762037, 0.93329744040966034] 

Epoch number, train and test accuracy  :   [0.72372553930726158, 0.66871227324008942] 

Epoch number, train and test loss      :   [0.75995648184487985, 0.92683118581771851] 

Epoch number, train and test accuracy  :   [0.72704886835674909, 0.67281492054462433] 

Epoch number, train and test loss      :   [0.74861955781315648, 0.91052477061748505] 

Epoch number, train and test accuracy  :   [0.73107226643451428, 0.67231190204620361] 

Epoch number, train and test loss      :   [0.73747052286946502, 0.91712185740470886] 

Epoch number, train and test accuracy  :   [0.73463647448739344, 0.6728542149066925] 

Epoch number, train and test loss      :   [0.72698959500290627, 0.92336003482341766] 

Epoch number, train and test accuracy  :   [0.73722579867340798, 0.67248235642910004] 

Epoch number, train and test loss      :   [0.71969311181889029, 0.92780447006225586] 

Epoch number, train and test accuracy  :   [0.73740224782810659, 0.67129908502101898] 

Epoch number, train and test loss      :   [0.71849010988723405, 0.93396638333797455] 

Epoch number, train and test accuracy  :   [0.73632618715596754, 0.66971391439437866] 

Epoch number, train and test loss      :   [0.72106433469195697, 0.93054975569248199] 

Epoch number, train and test accuracy  :   [0.73721961365189659, 0.66963332891464233] 

Epoch number, train and test loss      :   [0.718849778175354, 0.93013271689414978] 

Epoch number, train and test accuracy  :   [0.73815205346706303, 0.67015969753265381] 

Epoch number, train and test loss      :   [0.71525610186332877, 0.92774961888790131] 

Epoch number, train and test accuracy  :   [0.73733790114868514, 0.66859842836856842] 

Epoch number, train and test loss      :   [0.71633343225301693, 0.93185527622699738] 




"""









