import torch as tr
import numpy as np

tr_uniform = lambda a,b,shape: tr.FloatTensor(*shape).uniform_(a,b)


""" 
serial recall is a multi-trial task
a trial in the n-back task corresponds 
  to a probe in the serial recall task
  i.e. a trial is a sequence of probes
nprobes = setsize
"""

class SerialRecallTask():

  def __init__(self,sedim=2):
    self.sedim = sedim
    self.cedim = 2
    return None

  def gen_ep_data(self,ntrials,setsize):
    """
    episodes are multi-trial
    output: 
      C,X,Y [ntrials,setsize,edim]
      compatible with model input 
    """
    C = -np.ones([ntrials,setsize,self.cedim])
    S = -np.ones([ntrials,setsize,self.sedim])
    for trial in range(ntrials):
      S[trial] = np.abs(np.random.normal(0,.1,[setsize,self.sedim]))
    C = self.solar_cdrift_ep(ntrials,setsize)
    # convert np to tr
    C = tr.Tensor(C)
    S = tr.Tensor(S)
    return C,S,S

  # context drifts

  def cumsum_cdrift(self,stim):
    context_drift = tr.cumsum(stim,-1)
    return context_drift

  def linear_cdrift(self,nprobes,delta_M=1,cstd=0.5,cedim=5):
    """ 
    drifts ~N(1,self.cstd)
    returns a context embedding matrix [nprobes,cedim]
    """
    cstd = -np.ones([nprobes,self.cedim])
    v_t = np.random.normal(delta_M,self.cstd,self.cedim)
    for step in range(nprobes):
      delta_t = np.random.normal(delta_M,self.cstd,self.cedim)
      v_t += 0.5*delta_t
      cstd[step] = v_t
    return cstd

  def sphere_drift(self,nprobes,noise=0.05):
    """
    drift on a two dimensional sphere 
    returns C [nprobes,2]
    """
    th = np.pi/(2*nprobes) # drift rate
    R = np.array([[np.cos(th),-np.sin(th)],
                  [np.sin(th),np.cos(th)]])
    x = np.array([1,0]) # initial vector
    C = np.zeros([nprobes,2]) # context matrix
    for i in range(nprobes):
      x = R.dot(x) + np.random.normal(0,noise)
      x = x/np.linalg.norm(x)
      C[i] = x
    return C

  def solar_cdrift(self,nprobes,delta_M=.1,cstd=1):
    # sample initial point
    x_init = np.random.uniform(-1,1,[2])
    x_init /= np.linalg.norm(x_init)
    # init
    edim = x_init.shape[-1]
    arr = -np.ones([nprobes,edim])
    x = x_init
    # sample tstep context
    for step in range(nprobes):
      arr[step] = x
      delta_t = np.random.normal(delta_M,cstd,[edim])
      x += np.abs(delta_t)
    return arr

  def solar_cdrift_ep(self,ntrials,ntsteps):
    # deltas
    tstep_drift = lambda: (np.array([1,1]) + np.random.uniform(0,1))/30
    trial_shift = lambda: np.random.uniform(0.07,0.15)
    # drift shift loop
    context_arr = -np.ones([ntrials,ntsteps,2])
    th = 0
    for trial in range(ntrials):
      th += trial_shift()
      context_t = np.array([np.cos(th),np.sin(th)])
      for tstep in range(ntsteps):
        context_arr[trial,tstep] = context_t
        context_t += tstep_drift()
    return context_arr



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

