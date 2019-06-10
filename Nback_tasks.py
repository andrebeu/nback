import torch as tr
import numpy as np

tr_uniform = lambda a,b,shape: tr.FloatTensor(*shape).uniform_(a,b)


"""
This is the PM task used in WM+EM model
set num_pm_trials to 0 to make it an n-back task.
"""

class NbackTask_Basic():

  def __init__(self,nback,ntokens,edim,seed):
    """ 
    """
    np.random.seed(seed)
    tr.manual_seed(seed)
    self.nback = nback
    self.ntokens = ntokens
    self.edim = edim
    self.randomize_emat()
    return None

  def gen_seq(self,ntrials=20):
    """
    if pm_trial_position is not specified, they are randomly sampled
      rand pm_trial_position for training, fixed for eval
    """
    # generate og stim
    seq = np.random.randint(0,self.ntokens,ntrials)
    X = seq
    # form Y 
    Xroll = np.roll(X,self.nback)
    Y = (X == Xroll).astype(int) # nback trials
    return X,Y

  def embed_seq(self,X_seq,Y_seq):
    """ 
    takes 1-D input sequences
    returns 
      X_embed `(time,batch,edim)`[torch]
      Y_embed `(time,batch)`[torch]
    """
    # take signal_dim (time,edim_signal_dim)
    X_embed = self.emat[X_seq] 
    # include batch dim   
    X_embed = tr.unsqueeze(X_embed,1)
    Y_embed = tr.unsqueeze(tr.LongTensor(Y_seq),1)
    return X_embed,Y_embed

  def randomize_emat(self):
    self.emat = tr_uniform(-1,0,[self.ntokens,self.edim])


""" 
Nback with context drift
used for pure EM
"""

class NbackTask_PureEM():

  def __init__(self,nback=1,ntokens=3,cstd=0.3,cedim=2,sedim=2):
    """ 
    """
    self.nback = nback
    self.ntokens = ntokens
    self.cedim = cedim
    self.sedim = sedim
    self.cstd = cstd
    self.genseq = self.genseq_balanced
    self.sample_semat()
    return None

  # wrappers 

  def process_seq(self,seq):
    """ 
    takes 1D seq [np]
    returns 1D T,X,Y [np]
    """
    X = seq
    seqroll = np.roll(seq,self.nback)
    Y = (seqroll==seq).astype(int)
    Y[:self.nback] = 0
    T = np.arange(len(seq))
    return T,X,Y

  def seq2data(self,seq):
    """ 
    takes 1D seq
      embeds using 
        new context drift
        class stim emat
    returns embedded
      context,stim
    output of this function should 
      be compatible with input to model
    """
    T,X,Y = self.process_seq(seq)
    cemat = self.sample_cdrift(len(seq))
    context = tr.Tensor(cemat[T])
    stim = self.semat[X]
    Y = tr.LongTensor(Y).unsqueeze(0)
    return context,stim,Y

  def gen_ep_data(self,ntrials):
    """ top wrapper
    randomly generates an episode 
      according to class default settings
    output of this function should 
      be compatible with input to model
    """
    self.sample_semat()
    seq = self.genseq(ntrials)
    context,stim,Y = self.seq2data(seq)
    return context,stim,Y

  # sequence generators

  def genseq_naive(self,ntrials):
    """ naive sequence generator
    random sequence each token with equal probability
    """
    seq = np.random.randint(0,self.ntokens,ntrials)
    return seq

  def genseq_balanced(self,ntrials):
    """ if number of tokens is large, 
    this function balances probability of trial types
    """
    pr_nback = .6
    seq = -np.ones(ntrials)
    seq[:self.nback] = np.random.randint(0,self.ntokens,self.nback)
    for trial in range(self.nback,ntrials):
      nback_stim = seq[trial-self.nback]
      if np.random.binomial(1,pr_nback):
        stim = nback_stim
      else: 
        false_tokens = list(np.arange(self.ntokens))
        false_tokens.remove(nback_stim)
        stim = np.random.choice(false_tokens)
      seq[trial] = stim
    seq.astype(int)
    return seq

  # embedding matrices

  def sample_cdrift(self,ntrials,delta_M=1):
    """ 
    drifts ~N(1,self.cstd)
    returns a context embedding matrix [ntrials,cedim]
    """
    cstd = -np.ones([ntrials,self.cedim])
    v_t = np.random.normal(delta_M,self.cstd,self.cedim)
    for step in range(ntrials):
      delta_t = np.random.normal(delta_M,self.cstd,self.cedim)
      v_t += 0.5*delta_t
      cstd[step] = v_t
    return cstd

  def sample_semat(self):
    self.semat = tr.randn(self.ntokens,self.sedim)
    return None


class SerialRecall():
  def __init__(self,sedim=10,cedim=5,cstd=0.5):
    self.sedim = sedim
    self.cedim = cedim
    self.cstd = cstd
    return None

  def gen_ep_data(self,ntrials):
    """ 
    output [context,X,Y] compatible with model input [time,edim]
    """
    context_drift = tr.Tensor(self.sample_cdrift(ntrials))
    # stim = tr.randn(ntrials,self.sedim)
    stim = tr.Tensor(np.random.uniform(0,1,[ntrials,self.sedim]))
    context_drift = tr.cumsum(stim,0)
    return context_drift,stim,stim 

  # embedding matrices

  def sample_cdrift(self,ntrials,delta_M=1):
    """ 
    drifts ~N(1,self.cstd)
    returns a context embedding matrix [ntrials,cedim]
    """
    cstd = -np.ones([ntrials,self.cedim])
    v_t = np.random.normal(delta_M,self.cstd,self.cedim)
    for step in range(ntrials):
      delta_t = np.random.normal(delta_M,self.cstd,self.cedim)
      v_t += 0.5*delta_t
      cstd[step] = v_t
    return cstd

