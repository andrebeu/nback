import sys

import numpy as np
import torch as tr

from Nback_tasks import NbackTask_Basic
from Nback_models import PureWM

seed = int(sys.argv[1])
np.random.seed(seed)
tr.manual_seed(seed)

ntokens = 3
edim = 50
stsize = 10

nback1 = 1
task1 = NbackTask_Basic(nback1,ntokens,edim,seed)
net1 = PureWM(edim,stsize,seed)

nback2 = 3
task2 = NbackTask_Basic(nback2,ntokens,edim,seed)
net2 = PureWM(edim,stsize,seed)

train_epochs = 100000

## train and eval funs

def forward_prop(net,task,seqlen=20):
  """
  used both for training and eval
  returns 
    ep_loss(0)[tr],ep_score(tsteps)[np]
  """
  # loss op
  lossop = tr.nn.CrossEntropyLoss()
  # generate data
  x_seq,y_seq = task.gen_seq(seqlen)
  x_embeds,ytarget = task.embed_seq(x_seq,y_seq)
  # forward prop
  yhat = net(x_embeds)
  # collect loss through time
  yhat = yhat[task.nback:]
  ytarget = ytarget[task.nback:]
  ep_acc,ep_loss = 0,0
  ep_score = -np.ones(len(yhat))
  for tstep,(yh,yt) in enumerate(zip(yhat,ytarget)):
    ep_loss += lossop(yh,yt)
    ep_score[tstep] = yt==tr.argmax(tr.softmax(yh,1))
  return ep_loss,ep_score


def cloop_train(net,task,nepochs=1000,thresh=.99):
  # optiop
  optiop = tr.optim.Adam(net.parameters(), lr=0.0005)
  # loop
  tr_seqlen = 20 + task.nback
  train_acc = -np.ones(nepochs)
  nsets = 0
  for ep in range(nepochs):
    ep_loss,ep_score = forward_prop(net,task,seqlen=tr_seqlen)
    train_acc[ep] = ep_score.mean()
    # bp and update
    optiop.zero_grad()
    ep_loss.backward()
    optiop.step()
    # randomize
    if train_acc[ep] > thresh:
      task.randomize_emat()
      nsets+=1
  return train_acc,nsets

def nsets_train(net,task,target_nsets,thresh=.99):
  # optiop
  optiop = tr.optim.Adam(net.parameters(), lr=0.0005)
  # loop
  trseqlen = 20 + task.nback
  nsets,nepochs = 0,0
  while nsets <= target_nsets:
    nepochs+=1
    # fp and eval
    ep_loss,ep_score = forward_prop(net,task,seqlen=trseqlen)
    ep_acc = ep_score.mean()
    # bp and update
    optiop.zero_grad()
    ep_loss.backward()
    optiop.step()
    # randomize
    if ep_acc > thresh:
      task.randomize_emat()
      nsets+=1
  return nepochs


def eval_(net,task,neps=500):
  eval_seqlen = 20+task.nback
  eval_score = -np.ones((neps,eval_seqlen-task.nback))
  for ep in range(neps):
    ep_loss,ep_score = forward_prop(net,task,eval_seqlen)
    eval_score[ep] = ep_score
  return eval_score


# closed loop training on first net
print('TRAINING1')
acc,target_nsets = cloop_train(net1,task1,nepochs=train_epochs,thresh=.99)
# train to target number of sets on second net
print('TRAINING2',target_nsets)
nepochs = nsets_train(net2,task2,target_nsets,thresh=.99)

## eval
print('EVAL')
eval_score1 = eval_(net1,task1,neps=5000)
eval_score2 = eval_(net2,task2,neps=5000)

rootdir = 'model_data/pureWM-control_train_sets'
fnameL = [
          'evalscore-nback_%i-seed_%i'%(nback1,seed),
          'evalscore-nback_%i-seed_%i'%(nback2,seed),
          'seed_%i-nsets'%(seed),
          'seed_%i-epochs'%(seed)
          ]
arrL = [eval_score1,eval_score2,target_nsets,nepochs]
for fname,arr in zip(fnameL,arrL):
  fpath = rootdir+'/'+fname 
  np.save(fpath,arr)
