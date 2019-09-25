import numpy as np
import h5py
from glob import glob
from random import shuffle
def get_npz_sample(path):
    npzfile_path = glob(path+'/*.npz')
    npzfile_path = sorted(npzfile_path)
    print(npzfile_path)
    list_sample = []
    list_label = []
    list_slice = []
    shuffle(npzfile_path)
    for npzfile in npzfile_path:
        # print('start reading data from', npzfile.split('/')[-1])
        file = np.load(npzfile)
        # file_name = npzfile.split('/')[-1]
        sample = file['sample']
        label = file['label']
        slice = file['slice']
        if sample.shape != (128,128,3):
            print(npzfile.split('/')[-1],"is skipped because dimension incorrect")
            print(sample.shape, label.shape, label)
            continue
        list_sample.append(sample)
        list_label.append(label)
        list_slice.append(slice)
    print('finish loading data from', path)
    return list_sample, list_label, list_slice

"""
Prepare you data, such as:
"""
train_path = '../data/DSA_patch/train'
test_path = '../data/DSA_patch/test'
# val_path = '../data/DSA_patch/val'
output_path = '../data/'
# list_sample_train, list_label_train, list_slice_train = get_npz_sample(train_path)
list_sample_test, list_label_test, list_slice_test = get_npz_sample(test_path)
# list_sample_val, list_label_val = get_npz_sample(val_path)
# x_train = np.array(list_sample_train)  # should be a numpy array
# y_train = np.array(list_label_train)  # should be a numpy array
# slc_train = np.array(list_slice_train)
x_test = np.array(list_sample_test)    # should be a numpy array
y_test = np.array(list_label_test)    # should be a numpy array
slc_test = np.array(list_slice_test)
# x_val = np.array(list_sample_val)    # should be a numpy array
# y_val = np.array(list_label_val)
#print(x_train.shape, y_train.shape)
# x_test = np.concatenate([x_test,x_val],axis =0)
# y_test = np.concatenate([y_test, y_val], axis = 0)
# train_data = h5py.File(output_path+'data_train.h5','w')
# train_data.create_dataset('sample', data=x_train.astype('float64'))
# train_data.create_dataset('label', data=y_train.astype('int64'))
# train_data.create_dataset('slice', data=slc_train)
# train_data.close()
test_data = h5py.File(output_path+'data_test.h5','w')
test_data.create_dataset('sample', data=x_test.astype('float64'))
test_data.create_dataset('label', data=y_test.astype('int64'))
test_data.create_dataset('slice', data=slc_test)
test_data.close()
#val_data = h5py.File(output_path+'data_val.h5','w')
#val_data.create_dataset('sample', data=x_val.astype('float64'))
#val_data.create_dataset('label', data=y_val.astype('int64'))
#val_data.close()
