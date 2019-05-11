import sys

import numpy as np
import torch as tr

from Nback_tasks import NbackTask_PureEM
from Nback_models import PureEM


seed = int(sys.argv[1])
nback = int(sys.argv[2])
ntokens = int(sys.argv[3])

np.random.seed(seed)
tr.manual_seed(seed)

# model
cdim = 5
sedim = 10
stsize = 25
mthresh = 95
# task
cdrift = 5
train_neps = 100000

task = NbackTask_PureEM(nback=nback,ntokens=ntokens,cdrift=cdrift/10,cdim=cdim,sedim=sedim)
net = PureEM(indim=cdim+sedim,stsize=stsize,mthresh=mthresh/100)


model_fpath = "model_data/pureEM-sweep2/pureEM_%i-mthresh_%i-nback_%i-ntokens_%i-cdrift_%i-seed_%i"%(
                                          stsize,mthresh,nback,ntokens,cdrift,seed)
print(model_fpath)


def run_model(net,neps,gen_data_fn,training=False,verb=False):
  """ gen_data_fn: callable that generates a trial of data (embedded)
  """
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  # data generating option:
  C,X,Y = gen_data_fn()
  seqlen = len(C)
  # loop
  score = -np.ones([neps,seqlen])    
  for ep in range(neps):
    if ep%(neps/10)==0:
      if verb:print(ep/neps)
    # gen stim
    context,stim,ytarget = gen_data_fn()
    # forward prop
    yhat = net(context,stim)
    score[ep] = tr.argmax(tr.softmax(yhat,-1),-1).squeeze() == ytarget
    if training:
      for tstep in range(seqlen):
        loss = lossop(yhat[tstep],ytarget[:,tstep])
        optiop.zero_grad()
        loss.backward(retain_graph=True)
        optiop.step()
  return score


gen_rand_trials = lambda seqlen: lambda: task.gen_ep_data(seqlen)

train_score = run_model(net,train_neps,
                gen_data_fn=gen_rand_trials(10),
                training=True,verb=True)

eval_score = run_model(net,neps=500,
                       gen_data_fn=gen_rand_trials(15),
                       training=False)

np.save(model_fpath+'-train_score',train_score)
np.save(model_fpath+'-eval_score',eval_score)
tr.save(net.state_dict(),model_fpath+'-model.pt')
