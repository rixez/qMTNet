"""
Codes for testing qMTNet models 

Author: Huan Minh Luu
Magnetic Resonance Imaging Laboratory
KAIST
luuminhhuan@kaist.ac.kr
"""
import argparse
from keras.models import load_model
import scipy.io as sio
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from utils import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

""" Parameters setup """
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# basic parameters
parser.add_argument('--data_dir', default='./data/sample_test_data.mat', help='path to data')
parser.add_argument('--data_mode', default='conv', help='type of data to use (conv or inter)')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store results')
parser.add_argument('--model_dir', default='./models/qMTNet_fit_conv.h5', help='directory of the model to test')
parser.add_argument('--model_type', default='qMTNet_fit', help='type of model to train (one of qMTNet_1, qMTNet_fit, qMTNet_acq)')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='results are saved here')

opt = parser.parse_args()
if opt.gpu_ids == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    
save_dir = os.path.join(opt.checkpoints_dir,opt.name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
""""""""""""""""""""""""""

""" Create the model """
model = load_model(opt.model_dir)

""" Data Loading """
data_mat = sio.loadmat(opt.data_dir)
input_data = data_mat['test_input_' + opt.data_mode]
if opt.model_type == 'qMTNet_1':
    input_data = input_data[:,[0,1,2,7,8,13]]
    
output_data = data_mat['test_output_' + opt.data_mode]

if opt.data_mode == 'conv':
    input_data = input_data[:,:,np.newaxis]
    output_data = output_data[:,:,np.newaxis]
""" Testing the model """
prediction = np.zeros_like(output_data)
max_val = np.array([11.2,20.1])
for i in range(input_data.shape[2]):
    start = time.time()
    net_out = model.predict(input_data[:,:,i])
    print('Elapsed time: {}s'.format(time.time() - start))
    net_out = net_out*max_val
    net_out[output_data[:,:,i] == 0] = 0
    error = np.mean(np.square(net_out - output_data[:,:,i]))
    print('Average error for this slice: {}'.format(error))
    prediction[:,:,i] = net_out
    
    """ Save a png file of prediction and ground truth """
    net_out = np.reshape(net_out,(128,128,2),order='F')
    gt = np.reshape(output_data[:,:,i],(128,128,2),order='F')
    
    save_png(gt,net_out,os.path.join(save_dir,'slice_{}_{}.png'.format(i+1,opt.data_mode)))
    


sio.savemat(os.path.join(save_dir,'network_output.mat'),{'prediction':prediction})









