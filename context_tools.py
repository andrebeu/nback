import numpy as np

def linear_cdrift(nsteps,init_offset=0,delta_M=.1,cstd=0.05,cedim=2):
  """ 
  drifts ~N(1,self.cstd)
  returns a context embedding matrix [nsteps,cedim]
  """
  context = -np.ones([nsteps,cedim])
  v_t = init_offset*np.ones([1,cedim]) + np.random.normal(delta_M,cstd,cedim)
  for step in range(nsteps):
    delta_t = np.random.normal(delta_M,cstd,cedim)
    v_t += delta_t
    context[step] = v_t
  return context

def multitrial_linear_cdrift(ntrials,nsteps,cedim=2):
  context = -np.ones([ntrials,nsteps,cedim])
  for trial in range(ntrials):
    context[trial] = linear_cdrift(nsteps,init_offset=trial,cedim=cedim)
  return context