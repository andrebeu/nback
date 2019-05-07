import torch as tr
import numpy as np


nback=1
ntrials=10+nback
ntokens_og=3
stsize = 30
context_edim = 5
context_drift_std = .5
stim_edim = 10
nepochs=50000
  

def emb(stim_t_int,context_t_int,stim_emat,context_emat):
  xt_s_emb = stim_emat[stim_t_int].unsqueeze(0)
  xt_c_emb = context_emat[context_t_int].unsqueeze(0)
  xt_emb = tr.cat([xt_s_emb,xt_c_emb],-1) 
  return xt_emb

  

score = -np.ones(nepochs)


for ep in range(nepochs):
  if ep%(nepochs/5)==0:
    print(ep/nepochs)
  # embeddings matrices
  context_emat = tr.Tensor(task.generate_context_drift(ntrials,context_edim,context_drift_std))
  stim_emat = tr.randn(ntokens_og,stim_edim)
  # epoch data
  T,X,Y = task.gen_episode_data()
  X = tr.LongTensor(X)
  Y = tr.LongTensor(Y)
  # initial lstm
  h_t,c_t = tr.zeros(2,1,stsize)
  # initialize memory
  memoryL = []
  for tstep in range(nback):
    xt_emb = emb(X[tstep],tstep,stim_emat,context_emat)
    memoryL.append(xt_emb)
  net.init_memory(memoryL)
  # unroll
  acc = 0
  for tstep in range(nback,ntrials):
    # embed input
    x_t_int = X[tstep]
    xt_emb = emb(X[tstep],tstep,stim_emat,context_emat)
    # fp
    yhat,h_k,c_k = net(xt_emb,h_t,c_t)
    net.encode(xt_emb)
    # bp
    ytarget = Y[tstep].unsqueeze(0)
    loss = lossop(yhat,ytarget)
    optiop.zero_grad()
    loss.backward(retain_graph=True)
    optiop.step()
    acc = tr.softmax(yhat,1) == ytarget 
  score[ep] = None
  




task = NbackTask_PureEM(ntokens_og,ntrials)
net = PureEM(nback,context_edim+stim_edim,stsize,seed,debug=False)

lossop = tr.nn.CrossEntropyLoss()
optiop = tr.optim.Adam(net.parameters(), lr=0.001)