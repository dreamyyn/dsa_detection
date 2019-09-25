from __future__ import division
import math
# import dicom
import numpy as np
import pdb
import os
import shutil
from glob import glob
# import nibabel as nib
import SimpleITK as sitk
import scipy as sci
import scipy.ndimage
from skimage.transform import rescale

def glob_filename_list(path,searchterm='*',ext=''):
  filename_list = []
  for filename in glob(path+searchterm+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list

# load a series of dicom
def load_dicom_series_array(folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image = sitk.GetArrayFromImage(image)
    return image

# load a single dicom
def load_dicom_array(name):
    reader = sitk.ImageFileReader()
    reader.SetFileName(name)
    image = reader.Execute()
    image = sitk.GetArrayFromImage(image)
    return image

# concatenate a volumes' dicom togethers
def concat_images(filenames, input_size=None):
    images = []
    for i, fn in enumerate(filenames):
        image = load_dicom_array(fn)
        image = np.squeeze(image, axis=0)
        if input_size is not None:
            pdb.set_trace()
            image = sci.misc.imresize(image, (input_size,input_size))
        images.append(image)
    images = np.stack(images, 2)
    return images

def mean_normalize_volume(image):
    mean = np.mean(image)
    image = image / mean
    return image

# drop drop_num slices
def drop_slices(image, drop_num):
    # crop front and back 4 slices
    if drop_num == 0:
        image_crop = image
    elif drop_num == 1:
        image_crop = image[:,:,1:]
    elif drop_num == 2:
        image_crop = image[:,:,1:]
        image_crop = image_crop[:,:,:-1]
    elif drop_num == 3:
        image_crop = image[:,:,2:]
        image_crop = image_crop[:,:,:-1]
    elif drop_num == 4:
        image_crop = image[:,:,2:]
        image_crop = image_crop[:,:,:-2]
    else:       # random drop slice from image
        image_crop = image[:,:,2:]
        image_crop = image_crop[:,:,:-2]
        image_num = image_crop.shape[2]
        rnd_idx = np.arange(image_num)
        np.random.shuffle(rnd_idx)
        rnd_idx = rnd_idx[:4-drop_num]
        rnd_idx = np.sort(rnd_idx)
        image_crop = image_crop[:,:,rnd_idx]
    return image_crop

# relatively uniformly select dim slices from image
def clip_slices(image, dim):
    # random drop slices to make it the times of dim
    slices = image.shape[2]
    sample_num = slices // dim
    sample_drop = slices - sample_num * dim
    image_drop = drop_slices(image, sample_drop)

    # random select a sample from the image_drop
    rnd_idx = np.random.randint(sample_num)
    sample_idx = np.arange(rnd_idx, image_drop.shape[2], sample_num)
    image_sample = image_drop[:,:,sample_idx]
    return image_sample


# select previous "offset" slices plus current slice plus "offset" slices after current slice, hence total slices: 2*offset + 1
def select_slices(image, slice, offset=0):
    # find out the total slice number
    slices = image.shape[2]
    if slices < 2 * offset + 1:
        print('Offset {} is too large for {} slices.'.format(offset, slices))
    slice_start = slice - offset
    slice_end = slice + offset
    if slice_start < 0 or slice_end >= slices:
        print('Selecting slice {} with offset {} is out of bound. Total number of slices {}. Skipping...'.format(slice, offset, slices))
        return None
    slice_idx = range(slice_start, slice_end+1)
    image_sample = image[:,:,slice_idx]
    return image_sample

# random select a positive sample
# first select dim slices, then augmentation, then use pts as center to crop a image of ps x ps
def random_positive_sample(image_sample, pt, roi, ps=128,resize_factor=[0.5,1.5]):
    '''
    :param image_sample: image
    :param pt: the coordinates of the center point where samples are generted from
    :param roi: the boundary where samples are generated from
    :param ps: patch size of each sample
    :return: generate sample with a little bit randomization.
    '''
    img_h = image_sample.shape[0]
    img_w = image_sample.shape[1]
    
    # augmentation
    # 1. add an offset to the positive center, max 0.2 offset
    offset_ratio = 0.2
    offset_w = np.random.randint(-int(ps*offset_ratio), int(ps*offset_ratio)+1)
    offset_h = np.random.randint(-int(ps*offset_ratio), int(ps*offset_ratio)+1)
    pt_w = pt[0] + offset_w
    pt_w = max(min(pt_w, img_w-1-ps//2), ps//2)
    pt_h = pt[1] + offset_h
    pt_h = max(min(pt_h, img_h-1-ps//2), ps//2)

    # 2. rotation, [-30, 30]
    # pad the image to h+ps, w+ps
    rotate_angle = 10
    image_sample_pad = np.pad(image_sample, ((ps//2,ps//2),(ps//2,ps//2),(0,0)),'constant')
    # crop the image size with 2ps x 2ps, then rotate
    image_sample_crop = image_sample_pad[pt_h-ps//2:pt_h+3*ps//2, pt_w-ps//2:pt_w+3*ps//2, :]
    rnd_angle = np.random.randint(-rotate_angle, rotate_angle+1)
    img_rotate = []
    for i in range(image_sample_crop.shape[2]):
        img_slice = image_sample_crop[:,:,i]
        img_slice = sci.ndimage.rotate(img_slice, rnd_angle)
        img_rotate.append(img_slice)
    image_rotate = np.stack(img_rotate, axis=2)
    r_h= image_rotate.shape[0]
    r_w= image_rotate.shape[1]
    # crop tp ps x ps
    image_sample_final = image_rotate[r_h//2-ps//2:r_h//2+ps//2,r_w//2-ps//2:r_w//2+ps//2,:]
    
    ## try to combine local and global images. resize the patch size, sample image, and then resize again.
    # if resize_factor != []:
    #     for factor in resize_factor:
    #         image_rescale = rescale(image_sample, factor, anti_aliasing=False)
        
        
        



    # 3. horizontal flip
    if np.random.rand() > 0.5:      # flip
        image_sample_final = np.fliplr(image_sample_final)

    # compute center ratio in whole image and in roi
    ratio_whole_w = pt_w / img_w
    ratio_whole_h = pt_h / img_h
    roi_min_w = roi[0][0]
    roi_min_h = roi[0][1]
    roi_max_w = roi[2][0]
    roi_max_h = roi[2][1]
    ratio_roi_w = (pt_w - roi_min_w) / (roi_max_w - roi_min_w)
    ratio_roi_h = (pt_h - roi_min_h) / (roi_max_h - roi_min_h)

    return image_sample_final, [pt_h, pt_w], [ratio_whole_h, ratio_whole_w, ratio_roi_h, ratio_roi_w]


# random select a negative sample
# first select dim slices, then select crop a image of ps x ps in roi
def random_negative_sample(image_sample, pts, roi, ps):
    roi_min_w = roi[0][0]
    roi_min_h = roi[0][1]
    roi_max_w = roi[2][0]
    roi_max_h = roi[2][1]
    # print(roi_min_w,roi_min_h,roi_max_w,roi_max_h)

    img_h = image_sample.shape[0]
    img_w = image_sample.shape[1]

    # find possible negative center
    refind = True
    while(refind):
        pt_h = np.random.randint(roi_min_h+ps//2, roi_max_h-ps//2)
        pt_w = np.random.randint(roi_min_w+ps//2, roi_max_w-ps//2)
        refind = False
        for pt in pts:
            if (pt_h-pt[1])*(pt_h-pt[1])+(pt_w-pt[0])*(pt_w-pt[0]) <= ps*ps//2:
                refind = True

    # crop around the center
    image_sample_final = image_sample[pt_h-ps//2:pt_h+ps//2, pt_w-ps//2:pt_w+ps//2, :]

    # horizontal flip
    if np.random.rand() > 0.5:      # flip
        image_sample_final = np.fliplr(image_sample_final)

    # compute center ratio in whole image and in roi
    ratio_whole_w = pt_w / img_w
    ratio_whole_h = pt_h / img_h
    ratio_roi_w = (pt_w - roi_min_w) / (roi_max_w - roi_min_w)
    ratio_roi_h = (pt_h - roi_min_h) / (roi_max_h - roi_min_h)

    return image_sample_final, [pt_h, pt_w], [ratio_whole_h, ratio_whole_w, ratio_roi_h, ratio_roi_w]

def visualize_patch(sample, pt, idx, sample_dir, name_tag = ''):
    # pdb.set_trace()
    # # visualize all slices
    # for i in range(sample.shape[2]):
    #     img_path = sample_dir + str(idx) + '_' + str(i) + '_' + str(pt[0]) + '_' + str(pt[1]) + name_tag +'.jpg'
    #     sci.misc.imsave(img_path, sample[:,:,i])
    # visualize only central slice
    img_path = sample_dir + str(idx) + '_' + str(pt[0]) + '_' + str(pt[1]) + '_' + name_tag + '.jpg'
    sci.misc.imsave(img_path, sample[:, :, 1])
