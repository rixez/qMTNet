"""
Codes for training various models 

Author: Huan Minh Luu
Magnetic Resonance Imaging Laboratory
KAIST
luuminhhuan@kaist.ac.kr
"""
import argparse
from model import make_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import scipy.io as sio
import numpy as np
import os
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

""" Parameters setup """
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# basic parameters
parser.add_argument('--data_dir', default='./data/sample_train_data.mat', help='path to data')
parser.add_argument('--data_mode', default='conv', help='type of data to use (conv or inter)')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')


# training parameters
parser.add_argument('--model_type', required=True, help='type of model to train (one of qMTNet_1, qMTNet_fit, qMTNet_acq)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=500, help='number of training epoch')

opt = parser.parse_args()

if opt.gpu_ids == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    
""""""""""""""""""""""""""

""" Create the model """
model = make_model(opt.model_type)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=opt.lr))
model.summary()

""" Data Loading """
data_mat = sio.loadmat(opt.data_dir)
input_data = data_mat['train_input_' + opt.data_mode]
if opt.model_type == 'qMTNet_1':
    input_data = input_data[:,[0,1,2,7,8,13]]
    
output_data = data_mat['train_output_' + opt.data_mode]
output_data = output_data/np.array([11.2,20.1])

data_size = input_data.shape[0]
train_idx = np.random.choice(data_size, size=int(0.9*data_size),replace=False)
valid_idx = [x for x in range(data_size) if x not in train_idx]

train_input = input_data[train_idx,:]
train_output = output_data[train_idx,:]
valid_input = input_data[valid_idx,:]
valid_output = output_data[valid_idx,:]

""" Setting up training """
exp_dir = os.path.join(opt.checkpoints_dir,opt.name)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

modelName = os.path.join(exp_dir,opt.model_type + '_' + opt.data_mode + '.h5') 
checkpoint = ModelCheckpoint(modelName, monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
early = EarlyStopping(monitor='val_loss',min_delta=0,patience=50,verbose=1,mode='auto')
reducelrplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25,verbose=1)
   	
start = time.time()
model.fit(train_input,train_output,batch_size=128,epochs=opt.epoch,validation_data=(valid_input,valid_output),callbacks=[checkpoint,early,reducelrplateau],verbose=1)
print('Elapsed time: {} minutes'.format((time.time()-start)/60))






