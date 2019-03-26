import tensorflow as tf
import numpy as np

""" 
it is important to choose appropriate initializers for the M_keys 
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

  def gen_episode(self):
    """ 
    """
    seq = self.genseq_balanced_flip()
    seqroll = np.roll(seq,self.nback)
    X = np.expand_dims(seq,0)
    Y = (seqroll==seq).astype(int)
    Y[:self.nback] = 0
    Y = np.expand_dims(Y,0)
    T = np.expand_dims(np.arange(self.ntrials),0)
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
      T,X,Y = self.gen_episode()
      L.append(Y.sum()/Y.shape[1])
    print("-- proportion of true trials: M=%.2f S=%.2f"%(np.mean(L),np.std(L)))


""" 
Feed forward network with an HD
"""


class PureEM():

  def __init__(self,nstim,ntrials,dim=25,stim_edim=25,context_edim=25,discount_rate=0.9,nback=NBACK):
  	# task
    self.nback = nback
    self.nstim = nstim
    self.ntrials = ntrials
    # model
    self.dim = dim
    self.stim_edim = stim_edim
    self.context_edim = context_edim
    self.discount_rate = discount_rate
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
      self.stim = tf.keras.layers.Dense(self.dim,activation='relu')(self.stim_embed)
      self.context = self.context_embed
      # response layer
      response_layer1 = tf.keras.layers.Dense(self.dim,activation='relu')
      response_layer2 = tf.keras.layers.Dense(2,activation=None)
      response_dropout = tf.layers.Dropout(.9)
      self.response_layer = lambda x: response_layer2(response_dropout(response_layer1(x)))
      # unroll
      self.response_logits = self.unroll_trial(self.stim,self.context)
      ## loss and optimization
      self.y_hot = tf.one_hot(self.y_ph[:,self.nback:],2)
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.y_hot,
                          logits=self.response_logits)
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
          initializer=tf.initializers.glorot_normal)
    print('semat untrainable')
    # lookup
    context_embed = tf.nn.embedding_lookup(self.context_emat_ph,self.trial_ph,name='context_embed')
    stim_embed = tf.nn.embedding_lookup(self.stim_emat,self.stim_ph,name='stim_embed')
    return context_embed,stim_embed

  def unroll_trial(self,stim,context):
    # pre-load memory matrix with nback items
    self.M_keys = stim[0,:self.nback,:] # NB online mode 
    self.M_values = context[0,:self.nback,:]
    respL = []
    for tstep in range(self.nback,self.ntrials):
      stim_t = stim[:,tstep,:]
      context_t = context[:,tstep,:]
      # retrieve memory using stim
      retrieved_context_t = self.retrieve_memory(stim_t)
      # compute response
      self.response_in = tf.concat([context_t,retrieved_context_t],axis=-1)
      response_t = self.response_layer(self.response_in)
      respL.append(response_t)
      # write to memory (concat new stim to bottom)
      self.M_keys = tf.concat([self.M_keys,stim_t],
                      axis=0,name='M_keys_write') # [memory_idx,dim]
      self.M_values = tf.concat([self.M_values,context_t],
                      axis=0,name='M_values_write')
    response_logits = tf.stack(respL,axis=1,name='response_logits')
    return response_logits

  def retrieve_memory(self,query,temp=6):
    """
    NB works in online mode 
      matmul operation cannot handle 3D tensors [batch,key,dim]
    """
    keys = self.M_keys
    values = self.M_values
    # setup
    softmax = lambda x: tf.exp(temp*x)/tf.reduce_sum(tf.exp(temp*x),axis=0) 
    discount_arr = [self.discount_rate**i for i in range(keys.shape[0])] 
    discount_arr.reverse()
    # form retrieval similarity vector
    query_key_sim = 1-tf.keras.metrics.cosine(query,keys)
    query_key_sim = softmax(query_key_sim*discount_arr)
    self.query_key_sim = query_key_sim
    # use similarity to form memory retrieval
    retrieved_memory = tf.matmul(tf.expand_dims(query_key_sim,0),values)
    self.retrieved_memory = retrieved_memory
    return retrieved_memory



