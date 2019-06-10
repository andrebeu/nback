import torch as tr
import numpy as np

""" 
embedding happens outside of models
models 
  take in [time,batch,edim]
  return [time,outdim]
"""

class PureEM(tr.nn.Module):
  def __init__(self,nback=1,indim=4,stsize=5,mthresh=.95,seed=132,debug=False):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    self.debug = debug
    # dimensions
    self.indim = indim
    self.stsize = stsize
    self.outdim = 2
    # params
    self.mthresh = mthresh
    self.nback = nback
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
      context `(time,batch,cedim)`
      stim `(time,batch,sedim)`
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
  
  def retrieve(self,query):
    return self.retrieve_sort_rand(query)

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
