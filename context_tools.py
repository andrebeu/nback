import numpy as np

## 2d latice

def square_cdrift(nsteps,trial_offset,timestep_delta=0.1,cstd=None):
  """ 
  two dimensional context
  first dimension specified by trial_offset
  second dimension drifts at timestep_delta 
  """

  context = -np.ones([nsteps,2])
  for step in range(nsteps):
    tstep_context = step*timestep_delta
    trial_context = trial_offset
    v_t = np.array([trial_context,tstep_context])
    context[step] = v_t
  return context

def multitrial_square_cdrift(ntrials,nsteps):
  context = context = -np.ones([ntrials,nsteps,2])
  for trial in range(ntrials):
    context[trial] = square_cdrift(nsteps,trial)
  return context


## linear
def linear_cdrift(nsteps,init_offset=0,delta_M=.1,cstd=0.05,cedim=2):
  """ 
  drift starts from 'init_offset'*ones
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
  """ 
  multi-trial drift in direction of ones
  """
  context = -np.ones([ntrials,nsteps,cedim])
  for trial in range(ntrials):
    context[trial] = linear_cdrift(nsteps,init_offset=trial,cedim=cedim)
  return context

