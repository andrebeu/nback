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

"""


class PureEM():

  def __init__(self,nstim,ntrials,dim=25,stim_edim=25,context_edim=25,nback=NBACK,debug=False):
    # task
    self.nback = nback
    self.nstim = nstim
    self.ntrials = ntrials
    self.debug = debug
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
      self.y_ph = tf.placeholder(name='true_y_ph',shape=[1,self.ntrials],dtype=tf.int32)
      self.y_hot = tf.one_hot(self.y_ph[:,self.nback:],2)
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.y_hot,logits=self.response_logits)
      print('Loss',self.train_loss)

      G = tf.train.AdamOptimizer(0.001).compute_gradients(self.train_loss)
      for g in G:
        for i in g:
          print(i)
        
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
    self.dropout = tf.placeholder(name='dropout_ph',shape=[],dtype=tf.float32)
    return None

  def input_embeds(self):
    """ 
    untrainable orthogonal stim
    placeholder context
    """
    self.context_emat_ph = tf.placeholder(
          name='context_emat_ph',
          shape=[self.ntrials,self.context_edim],
          dtype=tf.float32)
    print('randn stimemat')
    self.stim_emat = tf.get_variable(
          name='stim_emat',
          shape=[self.nstim,self.stim_edim],
          trainable=False,
          initializer=tf.initializers.random_normal)
    self.randomize_stim_emat = self.stim_emat.initializer
    # lookup
    self.trial_ph = tf.placeholder(name='trial_ph',shape=[1,self.ntrials],dtype=tf.int32)
    self.stim_ph = tf.placeholder(name='stim_ph',shape=[1,self.ntrials],dtype=tf.int32)
    context_embed = tf.nn.embedding_lookup(self.context_emat_ph,self.trial_ph,name='context_embed')
    stim_embed = tf.nn.embedding_lookup(self.stim_emat,self.stim_ph,name='stim_embed')
    return context_embed,stim_embed

  def unroll_trial(self,stim,context):
    # pre-load memory matrix with nback items
    self.Memory_stim = stim[0,:self.nback,:]
    self.Memory_context = context[0,:self.nback,:]
    lstm_output_L = []
    self.respinL = []
    self.qsimL = []
    self.retrieved_memoriesL = []
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # initialize lstm
      lstm_cell = tf.keras.layers.LSTMCell(self.dim)
      lstm_layer = tf.keras.layers.RNN(lstm_cell,return_sequences=False,return_state=False)
      init_state_var = tf.get_variable(name='init_state',shape=[1,self.dim],trainable=True,initializer=tf.initializers.random_uniform()) 
      init_state = lstm_cell.get_initial_state(init_state_var)
      # out
      lstm2logits_layer = tf.keras.layers.Dense(2,activation=None)
      tf.keras.layers.Dropout(self.dropout)
      # unroll trial
      for tstep in range(self.nback,self.ntrials):
        # if tstep > 0: cellscope.reuse_variables()
        # trial input
        stim_t = stim[:,tstep,:]
        context_t = context[:,tstep,:]
        # retrieve memory using stim
        retrieved_memory,qsim_t = self.retrieve_memory(stim_t,context_t,self.memory_thresh)
        self.qsimL.append(qsim_t)
        self.retrieved_memoriesL.append(retrieved_memory)

        ## LSTM readout
        """ variable length
        first input is concat([sitm_t,context_t])
        followed by a variable length sequence of memories
          each memory also a concat([stim_i,context_i])
        """
        lstm_in = tf.concat([
                    tf.concat([
                      tf.expand_dims(stim_t,1),
                      tf.expand_dims(context_t,1)],axis=-1),
                    retrieved_memory],axis=1)
        # feed
        lstm_output = lstm_layer(lstm_in,initial_state=init_state)
        lstm_output_L.append(lstm_output)
        ## write to memory (concat new stim to bottom)
        self.Memory_stim = tf.concat([self.Memory_stim,stim_t],
                              axis=0,name='Mkeys_write') 
        self.Memory_context = tf.concat([self.Memory_context,context_t],
                              axis=0,name='Mvalues_write')
      # output projection to logits space
      lstm_outputs = tf.stack(lstm_output_L,axis=1,name='response_logits')
      response_logits = lstm2logits_layer(lstm_outputs)
    return response_logits

  def retrieve_memory(self,query_stim,query_context,thresh):
    """
    retrieves memories whose similarity which exceedes threshold
      similarity computed as stim_sim + context_sim
    returns a list of memory vectors 
      for which the similarity of query-key exceeds threshold
      sorted by similarity
    """ 
    # compute query-memory similarity
    query_stim_sim = -tf.keras.metrics.cosine(query_stim,self.Memory_stim) 
    query_context_sim = -tf.keras.metrics.cosine(query_context,self.Memory_context) 
    query_sim = query_stim_sim + query_context_sim
    # re-sort memory based on similarity to current query
    sort_idx = tf.argsort(query_sim,axis=-1)
    query_sim_sorted = tf.gather(query_sim,sort_idx)
    Memory_sorted = tf.gather(
                      tf.concat(
                        [self.Memory_stim,self.Memory_context],axis=-1
                      ),sort_idx)
    # thresholded retrieval 
    retrieved_memory = tf.boolean_mask(Memory_sorted,query_sim_sorted>thresh,axis=0) # variable length
    retrieved_memory = tf.expand_dims(retrieved_memory,axis=0)
    return retrieved_memory,query_sim_sorted


