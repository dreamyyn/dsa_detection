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
# import pdb
# import openpyxl as px
import csv
from glob import glob
import pickle
from tools import *

src_data_path = '/data/yannanyu/dsa_data/'
# src_data_path = '../data/DSA_raw/'
dst_data_path = '../data/DSA_pkl/'
# subj_info_path = '../subject_info.xlsx'

if not os.path.exists(dst_data_path):
    os.makedirs(dst_data_path)

# label_xls = px.load_workbook(subj_info_path)
# sheet = label_xls.get_sheet_by_name(name='Sheet1')

# subj_list = [30001,30002]
subj_list = [30001, 30002, 30004, 30007, 30008, 30010, 30011, 30012, 30013, 30015,
            30017, 30018, 30020, 30021, 30022, 30023, 30024,30026,30029,30031,30032]
# subj_list = [30001, 30002, 30004, 30007, 30008, 30010, 30011, 30012, 30013, 30015,
            # 30017, 30018, 30020, 30021, 30022, 30023, 30024, 30026, 30029, 30031,
            # 30032, 30036, 30038, 30039, 30043, 30077, 30078, 30079, 30084, 30087, 30128]

for subj in subj_list:
    print('start procecssing ' + str(subj))
    subj_path = glob(src_data_path+str(subj)+'/*')[0]
    subj_series_path = glob(subj_path+'/*')
    print(subj_series_path)
    subj_series_path.sort()
    label_dict = {}
    image_dict = {}
    ann_dict = {}

    # for each series
    subj_series_name_list = []
    for subj_series in subj_series_path:
        subj_series_name = subj_series.split('/')[-1]       # pre, post, etc
        if subj_series_name.endswith('_old'):
            continue
        subj_series_name_list.append(subj_series_name)
        print(subj_series_name)
        # load annotation from csv(block points + ROI)
        csv_path = glob(subj_series+'/*.csv')
        if len(csv_path) == 0:
            ann_dict[subj_series_name+ '_c'] = None
        else:
            roi = []
            pts = []
            slc = []
            with open(csv_path[0]) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if reader.line_num == 1:    # title line
                        continue
                    if int(row[19]) == 1:       # block point
                        pts.append([int(float(row[23])), int(float(row[24]))])
                        slc.append(int(float(row[0])))
                    else:
                        roi.append([int(float(row[23])), int(float(row[24]))])
                        roi.append([int(float(row[28])), int(float(row[29]))])
                        roi.append([int(float(row[33])), int(float(row[34]))])
                        roi.append([int(float(row[38])), int(float(row[39]))])
            ann_dict[subj_series_name+ '_c'] = {'slc':slc, 'pts':pts, 'roi':roi}
            print('finish loading the annotations '+ subj_series_name+ '_c')

        # load images
        subj_series_c_path = subj_series + '/c/'
        if not os.path.exists(subj_series_c_path):
            images_c = None
        else:
            filenames_c = glob(subj_series_c_path+'*')
            filenames_c.sort()
            images_c = concat_images(filenames_c)

        subj_series_l_path = subj_series + '/l/'
        if not os.path.exists(subj_series_l_path):
            images_l = None
        else:
            filenames_l = glob(subj_series_l_path+'*')
            filenames_l.sort()
            images_l = concat_images(filenames_l)

        image_dict[subj_series_name+'_c'] = images_c    # name as post_c etc
        image_dict[subj_series_name+'_l'] = images_l
        print('finish loading the images ' + subj_series_name)


    # load labels
    # num = 2
    # while(True):
    #     if sheet.cell(column=1, row=num).value == None:
    #         print('No info of subject {}'.format(subj))
    #         break
    #     if sheet.cell(column=1, row=num).value == subj:     # match subject
    #         label_list = []             # post1, post2,..,pre1, pre2...
    #
    #         for j in range(7, 11):       # check post set
    #             if sheet.cell(column=j, row=num).value == None:
    #                 break
    #             label = str(sheet.cell(column=j, row=num).value)
    #             label_list.append(label)
    #
    #         for j in range(2, 6):
    #             if sheet.cell(column=j, row=num).value == None:
    #                 break
    #             label = str(sheet.cell(column=j, row=num).value)
    #             label_list.append(label)
    #         break
    #     else:
    #         num = num + 1
    #
    # for i, subj_series_name in enumerate(subj_series_name_list):
    #     label_dict[subj_series_name] = label_list[i]
    # print('finish loading labels')
    # pdb.set_trace()

    # save a subj's data
    # data = {'subj_series_name_list':subj_series_name_list,
    #         'label_dict':label_dict, 'image_dict':image_dict, 'ann_dict':ann_dict}
    data = {'subj_series_name_list':subj_series_name_list,
             'image_dict':image_dict, 'ann_dict':ann_dict}
    with open(dst_data_path+str(subj)+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(dst_data_path+str(subj)+'.npz','w') as file_input:
    #     np.savez_compressed(file_input, subj_series_name_list=subj_series_name_list,
    #                         label_dict=label_dict, image_dict=image_dict, ann_dict=ann_dict)

    print("finish saving " + str(subj))


print('finish data preprocessing!')
