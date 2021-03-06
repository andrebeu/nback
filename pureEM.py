import tensorflow as tf
import numpy as np

""" 
it is important to choose appropriate initializers for the M_keys 
  otherwise everything will be too similar on the similarity-based lookup 
"""

NBACK = 2


# NSTIM = 2
# NTRIALS = 3
# class ToyNBackTask1():
  
#   def __init__(self,nback=NBACK):
#     self.nback = nback
#     return None

#   def genseq(self,ntrials=NTRIALS):
#     seq = np.array([0,1])
#     np.random.shuffle(seq)
#     seq = np.concatenate([seq,[0]])
#     seqroll = seq == np.roll(seq,self.nback)
#     seqroll[:self.nback] = 0
#     T = np.expand_dims(np.arange(ntrials),0)
#     X = np.expand_dims(seq,0)
#     Y = np.expand_dims(seqroll.astype(int),0)
#     return T,X,Y





NSTIM = 3
NTRIALS = 4
class ToyNBackTask2():
  
  def __init__(self,nback=NBACK):
    self.nback = nback
    return None

  def genseq(self,ntrials=NTRIALS):
    seq1 = np.random.choice([0,1,2],2,replace=False)
    np.random.shuffle(seq1)
    seq2 = np.random.choice([0,1,2],2,replace=False)
    np.random.shuffle(seq2)
    seq = np.concatenate([seq1,seq2])
    seqroll = seq == np.roll(seq,self.nback)
    seqroll[:self.nback] = 0
    T = np.expand_dims(np.arange(ntrials),0)
    X = np.expand_dims(seq,0)
    Y = np.expand_dims(seqroll.astype(int),0)
    return T,X,Y


""" 
Feed forward network with an HD
"""

class PureEM():

  def __init__(self,nback=NBACK,nstim=NSTIM,ntrials=NTRIALS,dim=10):
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
      # init memory mat
      response_layer1 = tf.keras.layers.Dense(self.dim,activation='relu')
      response_layer2 = tf.keras.layers.Dense(2,activation=None)
      self.response_layer = lambda x: response_layer2(response_layer1(x))
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
    # pre-load memory matrix
    self.M_keys = stim[0,:self.nback,:] # NB online mode 
    self.M_values = context[0,:self.nback,:]
    yL = []
    for tstep in range(self.nback,self.ntrials):
      stim_t = stim[:,tstep,:]
      context_t = context[:,tstep,:]
      # retrieve memory using stim
      retrieved_context_t = self.retrieve_memory(stim_t)
      # compute response
      response_in = tf.concat([context_t,retrieved_context_t],axis=-1)
      response_t = self.response_layer(response_in)
      yL.append(response_t)
      # write to memory
      self.M_keys = tf.concat([self.M_keys,stim_t],
                      axis=0,name='M_keys_write') # [memory_idx,dim]
      self.M_values = tf.concat([self.M_values,context_t],
                      axis=0,name='M_values_write')
    response_logits = tf.stack(yL,axis=1,name='response_logits')
    return response_logits


  def retrieve_memory(self,query,sm_temp=10):
    """
    NB online works in online mode 
      matmul operation cannot handle 3D tensors [batch,key,dim]
    """
    keys = self.M_keys
    values = self.M_values
    # form retrieval similarity vector
    query_key_sim = 1-tf.keras.metrics.cosine(query,keys)
    sm_sim = lambda x: tf.exp(sm_temp*x)/tf.reduce_sum(tf.exp(sm_temp*x),axis=0)
    query_key_sim = sm_sim(query_key_sim)
    # use similarity to form memory retrieval
    retrieved_memory_ = tf.transpose(
                        tf.matmul(
                          values,tf.expand_dims(query_key_sim,0),
                          transpose_a=True,transpose_b=True
                        ))
    retrieved_memory = tf.matmul(tf.expand_dims(query_key_sim,0),values)
    return retrieved_memory

  def write_to_memory(self,trial_num,key,value):
    return None

  def get_memory_mats(self):
    """ {self.stim: trial_embed}
    """
    return None

