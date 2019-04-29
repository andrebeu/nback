import torch as tr
import numpy as np

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
    X = seq
    Y = (seqroll==seq).astype(int)
    Y[:self.nback] = 0
    T = np.arange(len(seq))
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
      L.append(Y.sum()/Y.shape[0])
    print("-- proportion of true trials: M=%.2f S=%.2f"%(np.mean(L),np.std(L)))

  def generate_context_drift(self,ntrials,context_edim,delta_std=.3):
    delta_mean = 1
    alpha = .5
    arr = -np.ones([ntrials,context_edim])
    v_t = np.random.normal(delta_mean,delta_std,context_edim)
    for step in range(ntrials):
      delta_t = np.random.normal(delta_mean,delta_std,context_edim)
      v_t += alpha*delta_t
      arr[step] = v_t
    return arr



class Net(tr.nn.Module):
  def __init__(self,edim=4,stsize=5,seed=132,debug=False):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    self.debug = debug
    # params
    self.stsize = stsize
    self.outdim = 2
    self.mthresh = .95
    self.nback = NBACK
    # memory
    self.EM = None
    # layers
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.cell = tr.nn.LSTMCell(edim,stsize)
    self.lstmRNN = tr.nn.LSTM(edim,stsize)
    self.ff_out = tr.nn.Linear(stsize,self.outdim)
    return None

  def forward(self,x_t,h_t,c_t):
    """ 
    input: 
      x_t `(batch,dim)`
      h_t `(batch,dim)`
      c_t `(batch,dim)`
    """
    # lstm_in = 
    memories = self.retrieve(x_t)
    lstm_in = tr.cat([x_t,memories],0).unsqueeze(1)
    # for lstm_in_t in lstm_in:
    #   h_t,c_t = self.cell(lstm_in_t,(h_t,c_t))
    lstm_output,(h_t,c_t) = self.lstmRNN(lstm_in)
    output_t = self.ff_out(lstm_output[-1])
    return output_t,h_t,c_t
  
  def init_memory(self,memory_list):
    """ 
    EM_matrix is `(item,memory_dim)`
    """
    self.EM = tr.cat(memory_list)

  def encode(self,memory):
    """
    memory `(1,memory_dim)`
    """
    self.EM = tr.cat([self.EM,memory],0)

  def retrieve(self,query):
    """ 
    query is checked against memory elements
    returns retrieved memory
    since I initialized EM using percept array
      will reordering EM in here mess with order in percept array?
    """
    # compute similarity of query to stored memories
    sim = (tr.cosine_similarity(self.EM,query,dim=-1) + 1).detach()/2
    sorted_sim,sort_idx = tr.sort(sim,descending=True)
    sorted_EM = self.EM[sort_idx]
    retrieve_idx = sorted_sim > self.mthresh
    memories = sorted_EM[retrieve_idx]
    return memories