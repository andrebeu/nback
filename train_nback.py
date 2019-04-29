from datetime import datetime as dt
from glob import glob as glob
import torch as tr
import numpy as np

from pureEM_nback import *

ntrials=5
ntokens_og=3
stsize = 20
context_edim = 5
stim_edim = 5
seed = np.random.randint(100)

task = NBackTask(ntokens_og,ntrials,flip_pr=.8)
net = Net(context_edim+stim_edim,stsize,seed,debug=False)

lossop = tr.nn.CrossEntropyLoss()
optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  
nepochs = 100

L = -np.ones([nepochs])
A = -np.ones([nepochs])
E = -np.ones([nepochs])

stim_emat = tr.randn(ntokens_og,stim_edim,requires_grad=True)

context = task.generate_context_drift(ntrials,context_edim)
context = tr.Tensor(context).unsqueeze(1) # embed

t0 = dt.now()
for ep in range(nepochs):
  print(ep)
  if ep%(nepochs/5)==0:
    t0 = dt.now()
    print(dt.now()-t0)
    print(ep/nepochs)
  print('a')
  optiop.zero_grad() 
  # generate data
  T,X,Y = task.gen_episode_data()
  ytarget = tr.LongTensor(Y).unsqueeze(1)[2:] # onehot
  stimuli = stim_emat[X].unsqueeze(1) # embed
  # forward prop
  print('b')
  yhat = net(stimuli,context)
  print('c')
  # collect loss through time
  loss,acc = 0,0
  for yh,yt in zip(yhat,ytarget):
    loss += lossop(yh,yt)
    acc += yt==tr.argmax(tr.softmax(yh,1))
  print('d')
  acc = acc.numpy()/(ntrials-2)
  # bp and update
  print('e',loss)
  loss.backward()
  print('f')
  optiop.step()
  print('g')
  epoch_loss = loss.item()
  L[ep] = epoch_loss
  A[ep] = acc
