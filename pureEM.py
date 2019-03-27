import tensorflow as tf
import numpy as np

""" 
it is important to choose appropriate initializers for the Mkeys 
  otherwise everything will be too similar on the similarity-based lookup 
  also, using softmax with higher temperature seems to help

moving the discount factor out of the softmax worsens performance

larger embed sizes easier to learn, indepth helps regardless of edim
"""

NBACK = 2


class NBackTask():

  def __init__(self,nstim,ntrials,nback=NBACK,flip_pr=.35):
    """ 
    flip_pr controls how many of the stimuli in the 
      sequence will be flipped into a positive response trial
      this param controls chance level
    """
    self.nback = nback
    self.nstim = nstim
    self.ntrials = ntrials
    self.flip_pr = flip_pr
    self.sim_chance()
    return None

  def gen_episode_data(self):
    """ 
    """
    seq = self.genseq_balanced_flip()
    T,X,Y = self.format_seq_into_dataset(seq)
    return T,X,Y

  def format_seq_into_dataset(self,seq):
    seqroll = np.roll(seq,self.nback)
    X = np.expand_dims(seq,0)
    Y = (seqroll==seq).astype(int)
    Y[:self.nback] = 0
    Y = np.expand_dims(Y,0)
    T = np.expand_dims(np.arange(len(seq)),0)
    return T,X,Y

  def genseq_balanced_flip(self):
    # generate sequence
    seq = np.random.randint(0,self.nstim,self.ntrials)
    # flip true trials 
    # adjusted_flip_pr = self.flip_pr - self.flip_pr*(1/self.ntrials)
    true_nback_idx = np.where(np.random.binomial(1,self.flip_pr,self.ntrials-self.nback))[0] + self.nback
    for idx in true_nback_idx:
      seq[idx] = seq[idx-self.nback]
    return seq

  def genseq_unbalanced(self):
    seq = np.random.randint(0,self.nstim,self.ntrials)
    return seq

  def sim_chance(self):
    L = []
    for i in range(1000):
      T,X,Y = self.gen_episode_data()
      L.append(Y.sum()/Y.shape[1])
    print("-- proportion of true trials: M=%.2f S=%.2f"%(np.mean(L),np.std(L)))


""" 
Feed forward network with an HD
"""


class PureEM():

  def __init__(self,nstim,ntrials,dim=25,stim_edim=25,context_edim=25,nback=NBACK):
    # task
    self.nback = nback
    self.nstim = nstim
    self.ntrials = ntrials
    # model
    self.dim = dim
    self.stim_edim = stim_edim
    self.context_edim = context_edim
    self.memory_thresh = .8
    # graph
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      ## inputs
      self.placeholders()
      self.context_embed,self.stim_embed = self.input_embeds()
      self.context = self.context_embed
      self.stim = self.stim_embed
      # self.context = tf.keras.layers.Dense(self.dim,activation='relu')(self.context_embed)
      # self.stim = tf.keras.layers.Dense(self.dim,activation='relu')(self.stim_embed)
      # unroll
      self.response_logits = self.unroll_trial(self.stim,self.context)
      ## loss and optimization
      self.y_hot = tf.one_hot(self.y_ph[:,self.nback:],2)
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.y_hot,logits=self.response_logits)
      self.minimizer = tf.train.AdamOptimizer(0.001).minimize(self.train_loss)
      ## eval
      self.response_sm = tf.nn.softmax(self.response_logits)
      self.response = tf.argmax(self.response_sm,axis=-1)
      ## extra
      self.sess.run(tf.global_variables_initializer())
    return None

  def reinitialize(self):
    """ reinitializes variables to reset weights"""
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
    return None

  def placeholders(self):
    self.trial_ph = tf.placeholder(name='trial_ph',shape=[1,self.ntrials],dtype=tf.int32)
    self.stim_ph = tf.placeholder(name='stim_ph',shape=[1,self.ntrials],dtype=tf.int32)
    self.y_ph = tf.placeholder(name='true_y_ph',shape=[1,self.ntrials],dtype=tf.int32)
    self.dropout = tf.placeholder(name='dropout_ph',shape=[],dtype=tf.float32)
    self.context_emat_ph = tf.placeholder(name='context_emat_ph',shape=[self.ntrials,self.context_edim],dtype=tf.float32)
    return None

  def input_embeds(self):
    self.stim_emat = tf.get_variable(
          name='stim_emat',
          shape=[self.nstim,self.stim_edim],
          trainable=False,
          initializer=tf.initializers.orthogonal)
    # lookup
    context_embed = tf.nn.embedding_lookup(self.context_emat_ph,self.trial_ph,name='context_embed')
    stim_embed = tf.nn.embedding_lookup(self.stim_emat,self.stim_ph,name='stim_embed')
    return context_embed,stim_embed

  def unroll_trial(self,stim,context):
    # pre-load memory matrix with nback items
    self.Mkeys = stim[0,:self.nback,:] # NB online mode 
    self.Mvalues = context[0,:self.nback,:]
    lstm_output_L = []
    self.respinL = []
    self.qksimL = []
    self.retrieved_contextsL = []
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # initialize lstm
      print('non-trainable zero init')
      init_state_var = tf.get_variable('init_state',
                          shape=[1,self.dim],trainable=True,
                          initializer=tf.initializers.glorot_normal(0,1)
                          ) 
      lstm_cell = tf.keras.layers.LSTMCell(self.dim)
      init_state = lstm_cell.get_initial_state(init_state_var)
      lstm_layer = tf.keras.layers.RNN(lstm_cell,return_sequences=False,return_state=False)
      ##
      retrieve_null = False
      feed_context_to_lstm = False
      feed_context_to_out = True
      ##
      # unroll trial
      for tstep in range(self.nback,self.ntrials):
        # if tstep > 0: cellscope.reuse_variables()
        # trial input
        stim_t = stim[:,tstep,:]
        context_t = context[:,tstep,:]
        # retrieve memory using stim
        retrieved_contexts,qksim_t = self.retrieve_memoryL(stim_t,self.memory_thresh,retrieve_null)
        self.qksimL.append(qksim_t)
        self.retrieved_contextsL.append(retrieved_contexts)
        # LSTM readout
        if feed_context_to_lstm:
          lstm_in = tf.concat([tf.expand_dims(context_t,1),retrieved_contexts],1)
        else:
          lstm_in = tf.concat([tf.zeros([1,1,self.context_edim]),retrieved_contexts],1)
        # lstm_in = retrieved_contexts
        lstm_output = lstm_layer(lstm_in,initial_state=init_state)
        lstm_output_L.append(lstm_output)
        # write to memory (concat new stim to bottom)
        self.Mkeys = tf.concat([self.Mkeys,stim_t],
                        axis=0,name='Mkeys_write') # [memory_idx,dim]
        self.Mvalues = tf.concat([self.Mvalues,context_t],
                        axis=0,name='Mvalues_write')
      # output projection to logits space
      lstm_outputs = tf.stack(lstm_output_L,axis=1,name='response_logits')
      if feed_context_to_out:
        response_in = tf.concat([lstm_outputs,context[:,self.nback:,:]],-1)
        response_in = tf.keras.layers.Dense(self.dim,activation='relu')(response_in)
      else:
        response_in = lstm_outputs
      response_logits = tf.keras.layers.Dense(2,activation=None)(response_in)
    return response_logits

  def retrieve_memoryL(self,query,qksim_thresh,retrieve_null):
    """
    returns a list of memory vectors 
      for which the similarity of query-key exceeds threshold

    """ 
    Mkeys = self.Mkeys
    Mvalues = self.Mvalues
    
    qksim_arr = -tf.keras.metrics.cosine(query,Mkeys) 
    
    memoryL = []
    zeros = tf.zeros([1,self.context_edim])
    for m_i in range(qksim_arr.shape[0]):
      if retrieve_null:
        memory_i = tf.cond(qksim_arr[m_i] > qksim_thresh, 
                      true_fn=lambda: Mvalues[m_i,:], 
                      false_fn=lambda: 0*Mvalues[m_i,:])
        memoryL.append(memory_i)
        retrieved_contexts = tf.stack(memoryL,axis=0)
      else:
        retrieved_contexts = tf.boolean_mask(Mvalues,qksim_arr>qksim_thresh,axis=0)

    retrieved_contexts = tf.expand_dims(retrieved_contexts,axis=0)
    return retrieved_contexts,qksim_arr


