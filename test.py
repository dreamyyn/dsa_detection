import numpy as np
import pdb
from keras.models import Sequential,load_model
from keras.layers import Dense, Flatten, Conv2D,MaxPooling2D
from keras.regularizers import L1L2
from keras import backend as K
import keras.metrics
import keras.losses
import keras.optimizers
from glob import glob
import os
import h5py
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session


read_path = '../data/'
test = h5py.File(read_path + 'data_test.h5','r')
x_test = test.get('sample')
x_test = np.array(x_test)
y_test = test.get('label')
y_test = np.array(y_test)
test.close()
val = h5py.File(read_path + 'data_val.h5','r')
x_val = val.get('sample')
x_val = np.array(x_val)
y_val = val.get('label')
y_val = np.array(y_val)
val.close()

batch_size = 16
epochs = 10
filepath_checkpoint = '../data/ckpt.ckpt'
filepath_model = '../data/model.json'

'''
setup gpu
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
