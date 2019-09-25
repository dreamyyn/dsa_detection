 # transfer DSA image from DICOM to npz, split c and l axis
 # subj_series_name = ['pre'],
 # label_dict = {'pre':'0', ...},
 # ann_dict = {'pre_c':{'pts':[[24,24]], 'roi':[[24,24],[24,124],[48,24],[48,124]]}},
 # image_dict = {'pre_c':array}

# import argparse
# from scipy import io as sio
# import numpy as np
import os
# import dicom
# import nibabel as nib
# import sys
import pdb
# import openpyxl as px
import csv
from glob import glob
import pickle
from tools0917 import *

src_data_path = '/Volumes/My Passport/StanfordRSL/dsa_preprocess_data/dsa_AP/'
dst_data_path = '../data/DSA_pkl/'

subj_list = glob_filename_list(src_data_path,"3*")
print(subj_list)

if not os.path.exists(dst_data_path):
    os.makedirs(dst_data_path)

# label_xls = px.load_workbook(subj_info_path)
# sheet = label_xls.get_sheet_by_name(name='Sheet1')

# subj_list = ['30130','30131','30132']


for subj in subj_list:
    print('start procecssing ' + str(subj))
    subj_path = glob(src_data_path + str(subj)+'/*')
    subj_path.sort()
    label_dict = {}
    image_dict = {}
    ann_dict = {}
    
    # for each series
    subj_series_name_list = []
    for subj_series in subj_path:
        if '.csv' in subj_series:
            continue
        subj_series_name = subj_series.split('/')[-1]       # pre, post, etc
        subj_series_name_list.append(subj_series_name)
        print(subj_series_name)
        # TODO: load annotation from csv(block points + ROI)
        csv_path = src_data_path + str(subj) +'/{}_{}.csv'.format(subj,subj_series_name)
        print(csv_path)

        if not os.path.exists(csv_path):
            ann_dict[subj_series_name+ '_c'] = None
        else:
            roi = []
            loc = []  # record occlusion (ROI) anatomical location i.e. M1 M2 ICA
            pts = []  # coordinates of occlusion ROI
            slc = []  # slice number of occlusion ROI
            ica = []  # record the coordinates of ICA bifurcation
            phs = []  # record where parenchymal phase end and venous phase start
            with open(csv_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if reader.line_num == 1:    # title line
                        continue
                    if int(row[19]) == 19:       # ROItype 19 -> point ROI. block point
                        pts.append([int(float(row[27])), int(float(row[28]))])   # Note that Horos and OsiriX format is different in colume No. [23] [24] for Horos
                        slc.append(int(float(row[0])))
                        loc.append(row[7])
                    elif int(row[19]) == 6:      # ROItype 6 -> rectangular ROI
                        roi.append([int(float(row[27])), int(float(row[28]))])
                        roi.append([int(float(row[32])), int(float(row[33]))])
                        roi.append([int(float(row[37])), int(float(row[38]))])
                        roi.append([int(float(row[42])), int(float(row[43]))])
                    elif int(row[19]) == 9:     # ROItype 6 -> circle ROI
                        mm_px_ratio = float(row[17])/float(row[15])
                        ica.append([int(float(row[8])*mm_px_ratio/10),int(float(row[9])*mm_px_ratio/10)])
                    elif int(row[19]) == 5:   # 5 -> line ROI
                        phase = int(float(row[0]))
                        phs.append(phase)
                    else:
                        print("unknown ROI type")
                        continue
                    
            ann_dict[subj_series_name+ '_c'] = {'slc':slc, 'pts':pts, 'roi':roi, 'ica':ica, 'phs':phs,'loc':loc}
            print('finish loading the annotations '+ subj_series_name+ '_c')
        # pdb.set_trace()
        # TODO: load images

        if not os.path.exists(subj_series):
            images_c = None
        else:
            filenames_c = glob(subj_series+'/*.dcm')
            filenames_c.sort()
            images_c = concat_images(filenames_c[:phase])
            print(len(filenames_c),images_c.shape)

        image_dict[subj_series_name+'_c'] = images_c    # name as post_c etc
        # image_dict[subj_series_name+'_l'] = images_l
        print('finish loading the images ' + subj_series_name)

    data = {'subj_series_name_list':subj_series_name_list,
             'image_dict':image_dict, 'ann_dict':ann_dict}
    with open(dst_data_path+str(subj)+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(dst_data_path+str(subj)+'.npz','w') as file_input:
    #     np.savez_compressed(file_input, subj_series_name_list=subj_series_name_list,
    #                         label_dict=label_dict, image_dict=image_dict, ann_dict=ann_dict)

    print("finish saving " + str(subj))


print('finish data preprocessing!')
