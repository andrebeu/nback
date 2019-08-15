import sys
import torch as tr
import numpy as np

from Nback_tasks import ItemRecognitionTask
from Nback_models import ItemRecognitionEM
import context_tools as ct


seed = int(sys.argv[1])
ntrials = int(sys.argv[2])
setsize = int(sys.argv[3])
emthresh = int(sys.argv[4])


stsize = 25
sedim = 10
cedim = 2
stimset = 150

model_fname = "LSTM_%i-emthresh_%i-ntrials_%i-setsize_%i-seed_%i-noiseless"%(
                stsize,emthresh,ntrials,setsize,seed)

task = ItemRecognitionTask(sedim,cedim,stimset,seed)
net = ItemRecognitionEM(sedim,cedim,stsize,emthresh,seed)


maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)


def run_model(net,task,neps,ntrials,setsize,training=True,verb=True):
  """ train and eval
  """
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)

  score = -np.ones([neps,ntrials])
  loss = -np.ones([neps])

  for ep in range(neps):
    # task.sample_stokens()
    # gen stim, forward prop
    C,S,ytarget = task.gen_exp_data(ntrials,setsize)
    yhat = net(C,S)
    # scoring, backprop
    score[ep] = (maxsoftmax(yhat) == ytarget).squeeze()
    ep_loss = 0
    for trial in range(ntrials):
      tr_loss = lossop(yhat[trial],ytarget[trial])
      ep_loss += tr_loss
    if training:
      optiop.zero_grad()
      ep_loss.backward(retain_graph=True)
      optiop.step()
    # record and print
    loss[ep] = ep_loss
    if verb and ep%(neps/10)==0:
      print(ep/neps,ep_loss.detach().numpy())
  return loss,score


## train, eval, save
neps_tr = 250000
tr_loss,tr_sc = run_model(net,task,neps_tr,ntrials,setsize,training=True,verb=True)
fpath = "model_data/IRsweep_noiseless/"+model_fname
np.save(fpath+"-trsc",tr_sc)



# for s in np.arange(1,5):
#   # path
#   neps = s*neps_tr
#   fpath = "model_data/IRsweep_noiseless/"+model_fname+'-tr_%i'%neps
#   print(fpath)
#   # train and eval
#   tr_loss,tr_sc = run_model(net,task,neps_tr,ntrials,setsize,training=True,verb=False)
#   ev_loss,ev_score = run_model(net,task,neps_ev,ntrials,setsize,training=False,verb=False)
#   # saving
#   np.save(fpath+"-trsc",tr_sc)
#   np.save(fpath+"-evscore",ev_score)
#   tr.save(net.state_dict(),fpath+'-model.pt')
  


