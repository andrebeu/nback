import tensorflow as tf
import numpy as np

""" 
it is important to choose appropriate initializers for the M_keys 
  otherwise everything will be too similar on the similarity-based lookup 
  also, using softmax with higher temperature seems to help

moving the discount factor out of the softmax worsens performance
"""

NBACK = 2


class NBackTask():
  
  def __init__(self,nstim,nback=NBACK):
    self.nback = nback
    self.nstim = nstim
    return None

  def genseq(self,ntrials):
    seq = np.random.randint(0,self.nstim,ntrials)
    seqroll = np.roll(seq,2)
    X = np.expand_dims(seq,0)
    Y = (seqroll==seq).astype(int)
    Y[:self.nback] = 0
    Y = np.expand_dims(Y,0)
    T = np.expand_dims(np.arange(ntrials),0)
    return T,X,Y


""" 
Feed forward network with an HD
"""


class PureEM():

  def __init__(self,nstim,ntrials,dim=25,nback=NBACK):
    self.nback = nback
    self.nstim = nstim
    self.ntrials = ntrials
    self.dim = dim
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      ## model inputs
      self.trial_ph,self.stim_ph,self.y_ph = self.setup_placeholders()
      self.trial_embed,self.stim_embed = self.get_input_embeds(self.trial_ph,self.stim_ph)
      self.context,self.stim = self.trial_embed,self.stim_embed
      # response layer
      response_layer1 = tf.keras.layers.Dense(self.dim,activation='relu')
      response_dropout = tf.layers.Dropout(.9)
      response_layer2 = tf.keras.layers.Dense(2,activation=None)
      self.response_layer = lambda x: response_layer2(response_dropout(response_layer1(x)))
      # unroll
      self.response_logits = self.unroll_trial(self.stim,self.context)
      self.y_hot = tf.one_hot(self.y_ph[:,self.nback:],2)
      ## loss and optimization
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


  def setup_placeholders(self):
    trial_ph = tf.placeholder(name='trial_ph',shape=[1,self.ntrials],dtype=tf.int32)
    stim_ph = tf.placeholder(name='stim_ph',shape=[1,self.ntrials],dtype=tf.int32)
    y_ph = tf.placeholder(name='true_y_ph',shape=[1,self.ntrials],dtype=tf.int32)
    self.dropout = tf.placeholder(name='dropout_ph',shape=[],dtype=tf.float32)
    return trial_ph,stim_ph,y_ph


  def get_input_embeds(self,trial_ph,stim_ph):
    # setup emat
    self.trial_emat = tf.get_variable(
          name='trial_emat',
          shape=[self.ntrials,self.dim],
          trainable=True,
          initializer=tf.initializers.glorot_normal) 
    self.stim_emat = tf.get_variable(
          name='stim_emat',
          shape=[self.nstim,self.dim],
          trainable=True,
          initializer=tf.initializers.glorot_normal)
    # lookup
    trial_embed = tf.nn.embedding_lookup(self.trial_emat,self.trial_ph,name='trial_embed')
    stim_embed = tf.nn.embedding_lookup(self.stim_emat,self.stim_ph,name='stim_embed')
    return trial_embed,stim_embed


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


  def retrieve_memory(self,query,temp=6,discount_rate=0.9,discount_type='decaying'):
    """
    NB works in online mode 
      matmul operation cannot handle 3D tensors [batch,key,dim]
    """
    keys = self.M_keys
    values = self.M_values
    # form retrieval similarity vector
    query_key_sim = 1-tf.keras.metrics.cosine(query,keys)
    softmax = lambda x: tf.exp(temp*x)/tf.reduce_sum(tf.exp(temp*x),axis=0)
    # tanh = lambda x: tf.tanh(temp*x)/tf.reduce_sum(tf.tanh(temp*x),axis=0)
    # tanh = lambda x: tf.tanh(temp*x)
    if discount_type=='nback':
      discount_arr = [discount_rate**np.abs(i-1) for i in range(keys.shape[0])] 
    elif discount_type=='decaying': 
      discount_arr = [discount_rate**i for i in range(keys.shape[0])] 
    discount_arr.reverse()
    print(query)
    print(keys)
    print(discount_arr)
    query_key_sim = softmax(query_key_sim*discount_arr)
    # query_key_sim = tanh(query_key_sim*discount_arr)
    self.query_key_sim = query_key_sim
    # use similarity to form memory retrieval
    retrieved_memory = tf.matmul(tf.expand_dims(query_key_sim,0),values)
    self.retrieved_memory = retrieved_memory
    return retrieved_memory



