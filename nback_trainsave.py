## NBACK_CLASSIFICATION BRANCH 
import sys,os
from glob import glob as glob
import numpy as np
from nback import *


embed_size = int(sys.argv[1])
nback = 3
stsize = 50

# depth = 50
# num_epochs = 100000
# epochs_per_session = 500

depth = 5
num_epochs = 1000
epochs_per_session = 500

## initialize
ML = MetaLearner(stsize,depth,nback,nstim=3,embed_size=embed_size)
trainer = Trainer(ML)

## train / eval
train_loss,train_acc = trainer.train_loop(num_epochs,epochs_per_session)
eval_loss,eval_acc = trainer.eval_loop(500)

## save
model_name = 'state_%i-depth_%i-nback_%i-edim_%i'%(stsize,depth,nback,embed_size)
num_models = len(glob('models/sweep_edim/%s/*'%model_name)) 
model_dir = 'models/sweep_edim/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

# model
ML.saver_op.save(ML.sess,model_dir+'/final')
# train data
np.save(model_dir+"/train_loss-"+model_name,train_loss)
np.save(model_dir+"/train_acc-"+model_name,train_acc)
# eval data
np.save(model_dir+"/eval_loss-"+model_name,eval_loss)
np.save(model_dir+"/eval_acc-"+model_name,eval_acc)
