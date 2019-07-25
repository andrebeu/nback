import torch as tr
import numpy as np

# DBG = False


"""
Item-recognition (sternberg) task
"""

class ItemRecognitionEM(tr.nn.Module):

  """ 

  """

  def __init__(self,sedim,stsize,seed=155):
    super().__init__()
    self.mthresh = .88
    self.sedim = sedim
    self.stsize = stsize
    self.cedim = 2
    self.WM_lstm = tr.nn.LSTM(self.sedim+self.cedim,self.stsize)
    self.ff_out = tr.nn.Linear(self.stsize,self.sedim)
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    return None


  def forward_step(self,emL):
    """ 
    WM loop over candidate memories in emL
    single time step 
      takes emL [num_memories,cdim+sdim]
      returns yhat_tstep [1,sedim]
    """
    # if DBG: print('WM')
    h_t,c_t = self.initial_state
    emL = tr.unsqueeze(emL,1)
    yhat_tstep,(h_t,c_t) = self.WM_lstm(emL,(h_t.unsqueeze(0),c_t.unsqueeze(0)))
    yhat_tstep = yhat_tstep[-1]
    yhat_tstep = self.ff_out(yhat_tstep).relu()
    return yhat_tstep

  def forward_trial(self,context_trial):
    """ 
    input: trial context data 
      C: `[probe,cedim]
    loop over probes (timestep) in trial
      query EM with context_t 
        returns a variable length emL
          each em is a cat([context,stim])
        sorted according to similarity
          ensures current context always presented at beginning
      emL sequentially fed through WM lstm
    """
    setsize = context_trial.shape[0]
    yhat_trial = -tr.ones(setsize,self.sedim)
    for tstep in range(setsize):
      # if DBG: print('-probe',tstep)
      # retrieve EM
      em_query = context_trial[tstep]
      emL = self.retrieve_sort_sim(em_query)
      # if DBG: print('   len_emL',emL.shape)
      # feed WM
      yhat_tstep = self.forward_step(emL)
      yhat_trial[tstep] = yhat_tstep
      # if DBG: print('yh_ts:',yhat_tstep.shape)
    return yhat_trial

  def forward(self,context_arr,stim_arr):
    """
    input: episode data 
      S,C: `[trial,probe,edim]
    output: episode_yhat 
      Y: `[trial,probe,sdim]`
    loop over trials (sequence of probes of len `setsize`):
      encode trial data in EM
      forward prop trial

    """
    # task params
    ntrials = context_arr.shape[0]
    setsize = context_arr.shape[1]
    # initialize EM
    self.EM_K = tr.Tensor([])
    self.EM_V = tr.Tensor([])
    # unroll (multi_trial) episode 
    yhat_episode = -tr.ones(ntrials,setsize,self.sedim)
    # episode loop
    for trial in range(ntrials):
      # if DBG: print('\n--trial',trial)
      trial_context = context_arr[trial]
      trial_stim = stim_arr[trial]
      # encode EM
      self.encode(trial_context,trial_stim)
      # trial loop
      yhat_trial = self.forward_trial(trial_context) # [setsize,sedim]
      # if DBG: print('yh_tr:',yhat_trial.shape)
      yhat_episode[trial] = yhat_trial
    return yhat_episode


  def encode(self,trial_context,trial_stim):
    # if DBG: print('E')
    self.EM_K = tr.cat([self.EM_K,
                  trial_context
                  ],0)
    self.EM_V = tr.cat([self.EM_V,
                  tr.cat([trial_context,trial_stim],-1)
                  ],0)
    return None

  def retrieve_sort_sim(self,query):
    """ returns retrieved memories
    takes
      query `(1,indim)`
    returns 
      emL: `num_memories,memory_dim`
      memory_dim = sdim+cdim
    """
    # if DBG: print('R')
    # L2 distance
    dist = tr.nn.modules.distance.PairwiseDistance(2)
    sim = 1-dist(self.EM_K,query)
    # sort contents of EM according sim
    sorted_sim,sort_idx = tr.sort(sim,descending=True)
    sorted_EM_V = self.EM_V[sort_idx]
    # above threshold indices
    retrieve_idx = sorted_sim > self.mthresh
    # retrievals
    emL = sorted_EM_V[retrieve_idx]
    return emL





""" 
Serial recall
"""

class SerialRecallEM(tr.nn.Module):

  """ 
  at beginning `setsize` items encoded into EM
  each item consists of a (context,stimulus) tuple
  EM is { context : [context, stimulus] }
  for time-step in range(`setsize`):
    EM is queried with (true) current context
    EM returns emL (list) of above threshold items
    emL is fed through WM (lstm)
    WM-LSTM returns the next item
  """

  def __init__(self,sedim,stsize,seed=155):
    super().__init__()
    self.mthresh = .88
    self.sedim = sedim
    self.stsize = stsize
    self.cedim = 2
    self.WM_lstm = tr.nn.LSTM(self.sedim+self.cedim,self.stsize)
    self.ff_out = tr.nn.Linear(self.stsize,self.sedim)
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    return None


  def forward_step(self,emL):
    """ 
    WM loop over candidate memories in emL
    single time step 
      takes emL [num_memories,cdim+sdim]
      returns yhat_tstep [1,sedim]
    """
    # if DBG: print('WM')
    h_t,c_t = self.initial_state
    emL = tr.unsqueeze(emL,1)
    yhat_tstep,(h_t,c_t) = self.WM_lstm(emL,(h_t.unsqueeze(0),c_t.unsqueeze(0)))
    yhat_tstep = yhat_tstep[-1]
    yhat_tstep = self.ff_out(yhat_tstep).relu()
    return yhat_tstep

  def forward_trial(self,context_trial):
    """ 
    input: trial context data 
      C: `[probe,cedim]
    loop over probes (timestep) in trial
      query EM with context_t 
        returns a variable length emL
          each em is a cat([context,stim])
        sorted according to similarity
          ensures current context always presented at beginning
      emL sequentially fed through WM lstm
    """
    setsize = context_trial.shape[0]
    yhat_trial = -tr.ones(setsize,self.sedim)
    for tstep in range(setsize):
      # if DBG: print('-probe',tstep)
      # retrieve EM
      em_query = context_trial[tstep]
      emL = self.retrieve_sort_sim(em_query)
      # if DBG: print('   len_emL',emL.shape)
      # feed WM
      yhat_tstep = self.forward_step(emL)
      yhat_trial[tstep] = yhat_tstep
      # if DBG: print('yh_ts:',yhat_tstep.shape)
    return yhat_trial

  def forward(self,context_arr,stim_arr):
    """
    input: episode data 
      S,C: `[trial,probe,edim]
    output: episode_yhat 
      Y: `[trial,probe,sdim]`
    loop over trials (sequence of probes of len `setsize`):
      encode trial data in EM
      forward prop trial

    """
    # task params
    ntrials = context_arr.shape[0]
    setsize = context_arr.shape[1]
    # initialize EM
    self.EM_K = tr.Tensor([])
    self.EM_V = tr.Tensor([])
    # unroll (multi_trial) episode 
    yhat_episode = -tr.ones(ntrials,setsize,self.sedim)
    # episode loop
    for trial in range(ntrials):
      # if DBG: print('\n--trial',trial)
      trial_context = context_arr[trial]
      trial_stim = stim_arr[trial]
      # encode EM
      self.encode(trial_context,trial_stim)
      # trial loop
      yhat_trial = self.forward_trial(trial_context) # [setsize,sedim]
      # if DBG: print('yh_tr:',yhat_trial.shape)
      yhat_episode[trial] = yhat_trial
    return yhat_episode


  def encode(self,trial_context,trial_stim):
    # if DBG: print('E')
    self.EM_K = tr.cat([self.EM_K,
                  trial_context
                  ],0)
    self.EM_V = tr.cat([self.EM_V,
                  tr.cat([trial_context,trial_stim],-1)
                  ],0)
    return None

  def retrieve_sort_sim(self,query):
    """ returns retrieved memories
    takes
      query `(1,indim)`
    returns 
      emL: `num_memories,memory_dim`
      memory_dim = sdim+cdim
    """
    # if DBG: print('R')
    # L2 distance
    dist = tr.nn.modules.distance.PairwiseDistance(2)
    sim = 1-dist(self.EM_K,query)
    # sort contents of EM according sim
    sorted_sim,sort_idx = tr.sort(sim,descending=True)
    sorted_EM_V = self.EM_V[sort_idx]
    # above threshold indices
    retrieve_idx = sorted_sim > self.mthresh
    # retrievals
    emL = sorted_EM_V[retrieve_idx]
    return emL




"""
N-BACK TASK
"""

class PureEM(tr.nn.Module):
  def __init__(self,indim=4,stsize=5,mthresh=.95,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # dimensions
    self.indim = indim
    self.stsize = stsize
    self.outdim = 2
    # params
    self.mthresh = mthresh
    # memory
    self.EM = tr.Tensor([])
    # layers
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.lstmRNN = tr.nn.LSTM(indim,stsize)
    self.ff_out = tr.nn.Linear(stsize,self.outdim)
    return None

  def forward_step(self,x_t,h_t,c_t):
    """ 
    input: 
      x_t `(batch,dim)`
      h_t `(batch,dim)`
      c_t `(batch,dim)`
    """
    memories = self.retrieve(x_t)
    lstm_in = tr.cat([x_t,memories],0).unsqueeze(1) # (1+num_memories,1)
    lstm_output,(h_t,c_t) = self.lstmRNN(lstm_in)
    output_t = self.ff_out(lstm_output[-1])
    self.encode(x_t)
    return output_t,h_t,c_t

  def forward(self,context,stim):
    """ 
    input
      context `(time,cedim)`
      stim `(time,sedim)`
    returns
      yhat `time,batch,outdim` 
        unormalized
    """
    self.EM = tr.Tensor([])
    percept = tr.cat([context,stim],dim=-1)
    seqlen = len(percept)
    h_t,c_t = self.initial_state
    yhat = -tr.ones(seqlen,1,self.outdim)
    for tstep in range(seqlen):
      x_t = percept[tstep].unsqueeze(0)
      yhat_t,h_t,c_t = self.forward_step(x_t,h_t,c_t)
      yhat[tstep] = yhat_t
    return yhat

  def encode(self,memory):
    """
    memory `(1,memory_dim)`
    """
    self.EM = tr.cat([self.EM,memory],0)
    ## check if shufle, to ensure no info in order
    return None

  def retrieve(self,query):
    return self.retrieve_sort_rand(query)

  def retrieve_sort_rand(self,query):
    """ returns retrieved memories
    takes
      query `(1,indim)`
    returns 
      memories `num_memories,memory_dim`
      memory_dim = sedim+cdim
    """
    if len(self.EM)==0:
      return tr.Tensor([])
    # compute similarity of query to stored memories
    sim = (tr.cosine_similarity(self.EM,query,dim=-1) + 1).detach()/2
    sort_idx = np.random.permutation(np.arange(len(sim)))
    sorted_sim = sim[sort_idx]
    sorted_EM = self.EM[sort_idx]
    retrieve_idx = sorted_sim > self.mthresh
    memories = sorted_EM[retrieve_idx]
    return memories

  def retrieve_sort_sim(self,query):
    """ returns retrieved memories
    takes
      query `(1,indim)`
    returns 
      memories `num_memories,memory_dim`
      memory_dim = sedim+cdim
    """
    if len(self.EM)==0:
      return tr.Tensor([])
    # compute similarity of query to stored memories
    sim = (tr.cosine_similarity(self.EM,query,dim=-1) + 1).detach()/2
    sorted_sim,sort_idx = tr.sort(sim,descending=True)
    sorted_EM = self.EM[sort_idx]
    retrieve_idx = sorted_sim > self.mthresh
    memories = sorted_EM[retrieve_idx]
    return memories


class PureWM(tr.nn.Module):

  def __init__(self,indim,stsize,seed):
    self.stsize = stsize
    super().__init__()
    self.in2cell = tr.nn.Linear(indim,indim) # in2cell
    self.in2cell_relu = tr.nn.ReLU()
    # Main LSTM CELL
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.lstmcell = tr.nn.LSTMCell(indim,stsize)
    # outproj
    self.cell2out = tr.nn.Linear(stsize,stsize)
    self.cell2out_relu = tr.nn.ReLU()
    self.out2logits = tr.nn.Linear(stsize,2)
    return None

  def forward(self,xstim):
    """ 
    input: xstim `(time,batch,embedding)`
    output: yhat `(time,batch,outdim)`
    """
    seqlen = xstim.shape[0]
    # inproj
    percept = self.in2cell(xstim)
    percept = self.in2cell_relu(percept)
    ## unroll
    lstm_outs = -tr.ones(seqlen,1,self.stsize)
    # initial state
    lstm_output,lstm_state = self.initial_state
    for t in range(seqlen):
      # compute cell prediction
      lstm_output,lstm_state = self.lstmcell(percept[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    ## outporj
    yhat = self.cell2out(lstm_outs)
    yhat = self.cell2out_relu(yhat)
    yhat = self.out2logits(yhat)
    return yhat
