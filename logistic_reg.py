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
import scipy as sci
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session

TRAIN = False

def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def bce_recall_accuracy(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred) + 0.5 * (1 - recall(y_true,y_pred))+ 0.5 * (1 - keras.metrics.binary_accuracy(y_true,y_pred))
'''
load h5 data
'''
read_path = '../data/'
train = h5py.File(read_path + 'data_train.h5','r')
x_train = train.get('sample')
x_train = np.array(x_train)
y_train = train.get('label')
y_train = np.array(y_train)
train.close()
test = h5py.File(read_path + 'data_test.h5','r')
x_test = test.get('sample')
x_test = np.array(x_test)
y_test = test.get('label')
y_test = np.array(y_test)
test.close()
# val = h5py.File(read_path + 'data_val.h5','r')
# x_val = val.get('sample')
# x_val = np.array(x_val)
# y_val = val.get('label')
# y_val = np.array(y_val)
# val.close()
batch_size = 16
epochs = 20
filepath_checkpoint = '../data/ckpt.ckpt'
filepath_model = '../data/model.json'
print('pos_train:',np.sum(y_train), 'total_sample: ',len(y_train))
print('pos_test:',np.sum(y_test), 'total_sample: ',len(y_test))
'''
setup gpu
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model_checkpoint = ModelCheckpoint(filepath_checkpoint,
                                   monitor='val_loss',
                                   save_best_only=True,verbose=1)

"""
Set up the logistic regression model
"""
# model = Sequential()
# model.add(Flatten())
# model.add(Dense(2,  # output dim is 2, one score per each class
#                 activation='sigmoid',
#                 kernel_regularizer=L1L2(l1=0.0, l2=0.1),
#                 input_dim=x_train.shape[1]))  # input dimension = number of features your data has
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10, batch_size = 128, validation_data=(x_val, y_val))
# score = model.evaluate(x_test,y_test,verbose = 0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
'''
CNN
'''
#save model
if TRAIN:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)

    model.compile(loss=bce_recall_accuracy,
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy', recall,precision])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[model_checkpoint], validation_data=(x_test, y_test))


    #test the last model
    score_lastmodel = model.evaluate(x_test, y_test, verbose=0)
    # pred =  model.predict(x_test)
    pdb.set_trace()
    # y_list = np.concatenate([y_test.reshape(-1,1),pred],axis=1)
    # np.savetxt('../data/test.txt',y_list, fmt = '%.5f')
    print('Test loss:', score_lastmodel[0])
    print('Test accuracy:', score_lastmodel[1])

    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)
    model.save(filepath_model[:-5]+'.h5')

else:

    # del model
    model = load_model(filepath_model[:-5]+'.h5',custom_objects={'recall': recall,'precision':precision,'bce_recall_accuracy':bce_recall_accuracy}) ## load the final model (h5 format)
    model.load_weights(filepath_checkpoint)
    score_bestmodel = model.evaluate(x_test, y_test, verbose=0)
    pred =  model.predict(x_test)
    y_list = np.concatenate([y_test.reshape(-1,1),pred],axis=1)
    np.savetxt('../data/test.txt',y_list, fmt = '%.5f')
    print('Test loss:', score_bestmodel[0])
    print('Test accuracy:', score_bestmodel[1])

    # visualization
    pdb.set_trace()
    vis_path = '../data/visualize_res/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.exists(vis_path+'pos/'):
        os.makedirs(vis_path+'pos/')
    if not os.path.exists(vis_path+'neg/'):
        os.makedirs(vis_path+'neg/')

    for i in range(x_test.shape[0]):
        if y_test[i] == 0:
            vis_sub_path = vis_path + 'neg/'
        else:
            vis_sub_path = vis_path + 'pos/'
        for j in range(x_test.shape[3]):
            sci.misc.imsave(vis_sub_path+str(pred[i][0])+'_'+ str(i).zfill(4)+'_'+str(j)+'.jpg', x_test[i,:,:,j])
