# import argparse
# from scipy import io as sio
import numpy as np
import os
# import dicom
# import nibabel as nib
# import sys
import pdb
# from glob import glob
import pickle
from tools0917 import *

npz_data_path = '../data/DSA_pkl/'
visualize_path = '../data/visualize/'



# TODO: adjust the parameter here
# c_dim = 7           # num of slices per sample
# l_dim = 7
patch_size = 128    # size of the sample
pos_num_per_set = 1    # num of selected positive sample
neg_num_per_set = 1    # num of selected negative sample
mode = 'train'
# subj_list = [30001,30002]
if mode == 'train':
    # subj_list = glob_filename_list('/Volumes/My Passport/StanfordRSL/dsa_preprocess_data/dsa_AP/', "3*")
    subj_list =  [30161]    # train
    dst_data_path = '../data/DSA_patch/train/'
if mode == 'test':
    subj_list = [30132] # test
    dst_data_path = '../data/DSA_patch/test/' # test

if not os.path.exists(dst_data_path):
    os.makedirs(dst_data_path)

if not os.path.exists(visualize_path):
    os.makedirs(visualize_path)

for subj in subj_list:
    # define save path
    visualize_subj_path = visualize_path + str(subj) + '/'
    if not os.path.exists(visualize_subj_path):
        os.makedirs(visualize_subj_path)

    # load data
    with open(npz_data_path+str(subj)+'.pickle', 'rb') as handle:
        data = pickle.load(handle)

    subj_series_name_list = data['subj_series_name_list']
    image_dict = data['image_dict']
    ann_dict = data['ann_dict']

    for subj_series_name in subj_series_name_list:
        dst_filename_prefix = dst_data_path + str(subj) + '_' + subj_series_name
        visualize_subj_series_path = visualize_subj_path + subj_series_name + '/'
        if not os.path.exists(visualize_subj_series_path):
            os.makedirs(visualize_subj_series_path)
        if not os.path.exists(visualize_subj_series_path+'pos/'):
            os.makedirs(visualize_subj_series_path+'pos/')
        if not os.path.exists(visualize_subj_series_path+'neg/'):
            os.makedirs(visualize_subj_series_path+'neg/')

        # process data for c
        image = image_dict[subj_series_name+'_c']
        #  TODO: check whether appropriate or not
        image_normalized = mean_normalize_volume(image)
        image_MIP = np.max(image_normalized,axis=2)
        ann = ann_dict[subj_series_name+'_c']
        pts = ann['pts']
        roi = ann['roi']
        slc = ann['slc']
        ica = ann['ica']
        phs = ann['phs']

        # TODO: check pts exist
        if len(pts) > 0:
            for i in range(len(pts)):
                cnt_pos = 0
                for r in range(pos_num_per_set):
                    # pdb.set_trace()
                    # select random slices
                    # image_sample = clip_slices(image, c_dim)
                    # select 3 slices around slc
                    image_sample = select_slices(image, slc[i], offset=1)
                    if image_sample is None:
                        continue
                    image_input = np.dstack((image_sample,image_MIP)) # add MIP to input
                    
                    positive_sample, positive_pt, position_ratio = random_positive_sample(image_input, pts[i], roi, patch_size)
                    with open(dst_filename_prefix+'_pos_slice{0}'.format(slc[i])+'_'+str(r)+'.npz','wb') as file_input:
                        np.savez_compressed(file_input, sample=positive_sample, pt=positive_pt, position_ratio=position_ratio, label=1, slice=slc[i])

                    cnt_pos += 1
                    # pdb.set_trace()
                    try:
                        visualize_patch(positive_sample, positive_pt, r, visualize_subj_series_path + 'pos/', name_tag = 'slice' + str(slc[i]))
                    except ValueError: #raised if empty
                        pass
                    print('generating positive patch, slice ' + str(i) + ' for '+ str(subj) + ', time point ' + subj_series_name)

        for i in range(1,image.shape[2]-1):
            cnt_neg = 0
            for r in range(neg_num_per_set):
                # first random select #dim slices
                # image_sample = clip_slices(image, c_dim)
                # select 3 slices around slc
                image_sample = select_slices(image, i, offset=1)
                if image_sample is None:
                    continue
                image_input = np.dstack((image_sample, image_MIP))  # add MIP to input
                negative_sample, negative_pt, position_ratio= random_negative_sample(image_input, pts, roi, patch_size)
                with open(dst_filename_prefix+'_neg_slice{0}'.format(i)+'_' + str(r)+'.npz','wb') as file_input:
                    np.savez_compressed(file_input, sample=negative_sample, pt=negative_pt, \
                                            position_ratio=position_ratio, label=0,slice=i)
                cnt_neg += 1
                try:
                    visualize_patch(negative_sample, negative_pt, r, visualize_subj_series_path+'neg/', name_tag = 'slice' + str(i))
                except ValueError: #raised if array empty
                    pass
                print('generating negative patch, slice ' + str(i) + ' for '+ str(subj) + ', time point ' + subj_series_name)


print('finish data preparation!')
