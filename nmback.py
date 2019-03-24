import tensorflow as tf
from datetime import datetime as dt
import numpy as np

BATCH_SIZE = 1

""" 
no preunroll: just backprop to begining
make task_flag vectors non-randomizable and trainable
"""

class NMBackTask():

  def __init__(self,nback,mback,nstim):
    """ assume m>n
    """
    self.nmbackL = [nback,mback]
    self.nstim = nstim
    return None

  def gen_seq(self,ntrials,task_flag):
    """
    task_flag can be 0 or 1 to indicate whether
      to perform nback or mback.
    returns X=[[x(t),y(t-t)],...] Y=[[[y(t)]]]
        `batch,time,step`
    """
    # sample either n or mback task
    nmback = self.nmbackL[task_flag]
    # compose sequnece
    seq = np.random.randint(2,self.nstim+2,ntrials)
    seq_roll = np.roll(seq,nmback)
    Xt = seq
    Yt = (seq==seq_roll).astype(int)
    Xt = np.append(task_flag,Xt)
    Yt = np.append(task_flag,Yt)
    X = np.expand_dims(Xt[:-1],0) 
    Y = np.expand_dims(Yt[:-1],0)
    return X,Y


class MetaLearner():

  def __init__(self,stsize,depth=None,nstim=3,edim=8):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.stsize = stsize
    # self.depth = depth 
    self.nstim = nstim
    self.edim = edim
    self.num_actions = 2
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      ## data feeding
      self.setup_placeholders()
      self.xbatch_id,self.ybatch_id,self.itr_initop = self.data_pipeline() 
      # embedding matrix and randomization op
      self.stim_emat = tf.get_variable('stimulus_embedding_matrix',[self.nstim,self.edim],
                        trainable=False,initializer=tf.initializers.random_normal(0,1)) 
      self.task_emat = tf.get_variable('task_embedding_matrix',[2,self.edim],
                        trainable=True,initializer=tf.initializers.random_normal(0,1)) 
      self.emat = tf.concat([self.task_emat,self.stim_emat],0)
      self.randomize_stim_emat = self.stim_emat.initializer
      ## inference
      self.xbatch = tf.nn.embedding_lookup(self.emat,self.xbatch_id,name='xembed') 
      self.yhat_unscaled,self.final_state = self.RNNinference_keras(self.xbatch) 
      self.ybatch_onehot = tf.one_hot(self.ybatch_id,self.num_actions) 
      ## train
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,
                          logits=self.yhat_unscaled)
      self.minimizer = tf.train.AdamOptimizer(0.001).minimize(self.train_loss)
      ## eval
      self.yhat_sm = tf.nn.softmax(self.yhat_unscaled) 
      # self.acc = tf.equal(self.ybatch_id,tf.argmax(self.yhat_sm,2,output_type=tf.int32))
      # self.acc = tf.metrics.accuracy(
      #             labels=self.ybatch_id,
      #             predictions=tf.argmax(self.yhat_sm,2))
      
      # other
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver(max_to_keep=None)
    return None

  def reinitialize(self):
    """ reinitializes variables to reset weights"""
    with self.graph.as_default():
      print('randomizing params')
      self.sess.run(tf.global_variables_initializer())
    return None
  
  # setup 

  def data_pipeline(self):
    """data pipeline
    returns x,y = get_next
    also creates self.itr_initop
    """
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,self.yph))
    dataset = dataset.batch(BATCH_SIZE)
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    itr_initop = iterator.make_initializer(dataset)
    xbatch,ybatch = iterator.get_next() 
    return xbatch,ybatch,itr_initop

  def setup_placeholders(self):
    """
    setup placeholders as instance variables
    """
    self.xph = tf.placeholder(tf.int32,
              shape=[None,None],
              name="xdata_placeholder")
    self.yph = tf.placeholder(tf.int32,
                  shape=[None,None],
                  name="ydata_placeholder")
    self.dropout_keep_pr = tf.placeholder(tf.float32,
                  shape=[],
                  name="dropout_ph")
    return None

  # inference

  def RNNinference_(self,xbatch):
    """ 
    depricated. keeping this here in case I need to analyze state trajectory in the future
    current keras implementation of RNN unroll does not allow outputting cell states
    """
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # setup RNN cell      
      # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
      #       self.stsize,dropout_keep_prob=self.dropout_keep_pr)
      ## input projection
      # [batch,depth,insteps,edim]
      xbatch = tf.layers.dense(xbatch,self.stsize,
                  activation=tf.nn.relu,name='inproj')
      # [batch,depth,insteps,stsize]
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      ## unroll 
      outputL = []
      for tstep in range(self.depth):
        if tstep > 0: cellscope.reuse_variables()
        output,state = cell(xbatch[:,tstep,:], state)
        outputL.append(output)
    # output projection
    outputs = tf.convert_to_tensor(outputL) 
    outputs = tf.transpose(outputs,(1,0,2)) 
    outputs = tf.layers.dense(outputs,self.stsize,
                  activation=tf.nn.relu,name='hidoutlayer')
    outputs = tf.layers.dense(outputs,self.num_actions,
                  activation=None,name='outproj')
    return outputs,state

  def RNNinference_keras(self,xbatch):
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      lstm_cell = tf.keras.layers.LSTMCell(self.stsize)
      init_state_var = tf.get_variable('init_state',[BATCH_SIZE,self.stsize],
                        trainable=True,initializer=tf.initializers.random_normal(0,1)) 
      init_state = lstm_cell.get_initial_state(init_state_var)
      lstm_layer = tf.keras.layers.RNN(
        lstm_cell,return_sequences=True,return_state=True)
      # layers
      dropout_inlayer = tf.keras.layers.Dropout(rate=(1-self.dropout_keep_pr))
      dropout_outlayer = tf.keras.layers.Dropout(rate=(1-self.dropout_keep_pr))
      dense_inlayer1 = tf.keras.layers.Dense(35,activation='relu')
      dense_inlayer2 = tf.keras.layers.Dense(self.stsize,activation='relu')
      dense_outlayer1 = tf.keras.layers.Dense(int(self.stsize/2),activation='relu')
      dense_outlayer2 = tf.keras.layers.Dense(self.num_actions,activation=None)
      # forward prop
      xbatch = dropout_inlayer(dense_inlayer1(xbatch))
      xbatch = dense_inlayer2(xbatch)
      lstm_outputs,final_output,final_state = lstm_layer(xbatch,initial_state=init_state)
      lstm_outputs = dropout_outlayer(lstm_outputs)
      outputs = dense_outlayer2(dense_outlayer1(lstm_outputs))
      print('dropout in and out')
    return outputs,final_state




class Trainer():

  def __init__(self,net,nback,mback,trials_per_episode=50):
    self.trials_per_episode = trials_per_episode
    self.net = net
    self.task = NMBackTask(nback,mback,nstim=self.net.nstim)
    return None

  def train_step(self,Xdata,Ydata,cell_state=None):
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_keep_pr:0.9,
                  }
    # initialize iterator
    self.net.sess.run([self.net.itr_initop],feed_dict)
    # update weights and compute final loss
    cell_st,_ = self.net.sess.run(
      [self.net.final_state,self.net.minimizer],feed_dict)
    return cell_st

  def train_close_loop(self,num_epochs,thresh=0.95):
    num_evals = num_epochs
    acc_arr = -1*np.ones([num_evals])
    k_arr = -1*np.ones([num_evals])
    nrand = 0
    for ep_num in range(num_epochs):
      # generate data
      task_flag = np.random.choice([0,1])
      Xdata,Ydata = self.task.gen_seq(self.trials_per_episode,task_flag=task_flag)
      self.train_step(Xdata,Ydata)
      # eval
      evalstep_acc = np.mean(self.eval_step(Xdata,Ydata))
      acc_arr[ep_num] = evalstep_acc
      k_arr[ep_num] = nrand
      if evalstep_acc >= thresh:
        self.net.sess.run(self.net.randomize_stim_emat)
        nrand += 1
      if ep_num%(num_epochs/20) == 0:
          print(ep_num/num_epochs,np.mean(evalstep_acc),'k=',nrand)
    return acc_arr,k_arr

  def train_loop(self,num_epochs,epochs_per_session):
    """
    """
    num_evals = num_epochs
    acc_arr = np.empty([num_evals])
    eval_idx = -1
    for ep_num in range(num_epochs):
      if ep_num%epochs_per_session == 0:
        # randomize embeddings
        self.net.sess.run(self.net.randomize_stim_emat)
      # # generate data
      task_flag = np.random.choice([0,1])
      Xdata,Ydata = self.task.gen_seq(self.trials_per_episode,task_flag=task_flag)
      # train step
      self.train_step(Xdata,Ydata)
      # printing
      if ep_num%(num_epochs/num_evals) == 0:
        eval_idx += 1
        evalstep_acc = self.eval_step(Xdata,Ydata)
        acc_arr[eval_idx] = np.mean(evalstep_acc)
        if ep_num%(num_epochs/20) == 0:
          print(ep_num/num_epochs,np.mean(evalstep_acc)) 
    return acc_arr

  def eval_step(self,Xdata,Ydata):
    ## setup
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_keep_pr:1.0,
                  }
    self.net.sess.run([self.net.itr_initop],feed_dict)
    ## eval
    step_yhat_sm,step_ybatch = self.net.sess.run(
                                        [
                                        self.net.yhat_sm,
                                        self.net.ybatch_onehot
                                        ],feed_dict)
    step_acc = step_yhat_sm.argmax(2) == step_ybatch.argmax(2)
    return step_acc

  def eval_loop(self,num_itr,trials_per_episode=None,task_flag=None):
    if trials_per_episode==None:
      trials_per_episode = self.trials_per_episode
    acc_arr = np.zeros([num_itr,trials_per_episode])
    for it in range(num_itr):
      self.net.sess.run(self.net.randomize_stim_emat)
      Xdata,Ydata = self.task.gen_seq(trials_per_episode,task_flag)
      step_acc = self.eval_step(Xdata,Ydata)
      acc_arr[it] = step_acc
    return acc_arr
