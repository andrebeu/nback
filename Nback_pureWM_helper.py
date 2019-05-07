import torch as tr
import numpy as np

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