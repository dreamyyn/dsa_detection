import pydicom
import os
from glob import glob
import numpy as np
# from skimage.filters import frangi, hessian
from scipy import ndimage
'''
purpose:
1. revert the gray scale of DSA. background-blk, vessel-white
2. apply median filters to remove noise
3. track vessel structures using frangi filter
'''
def glob_filename_list(path,searchterm='*',ext=''):
  filename_list = []
  for filename in glob(path+searchterm+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list


cohort_path = '/Volumes/My Passport/StanfordRSL/dsa_train_data/dsa_AP/'
output_path = '/Volumes/My Passport/StanfordRSL/dsa_preprocess_data/dsa_AP/'
pt_list = glob_filename_list(cohort_path)
# pt_list.sort()
print(pt_list)
for subj_id in pt_list:
  if 'OsiriX' in subj_id:
    continue
  dicom_list = glob(cohort_path + subj_id +'/*/*/*.dcm')
  print(subj_id, len(dicom_list))
  for dicom_img in dicom_list:
    print(dicom_img)

    # read dicom images
    ds = pydicom.dcmread(dicom_img)
    img = ds.pixel_array
    
    # revert image grayscale
    a = np.uint16(3000)   # only uint16 can be used in dicom
    # print('before revert:',img.shape, np.mean(img))
    # print(img[300:306,300:306])
    img[img>a] = a
    revert_img = a - img
    reverted_img = np.maximum(0, np.nan_to_num(revert_img, 0))
    # print('after revert:',reverted_img.shape, np.mean(reverted_img))
    # print(reverted_img[300:306,300:306])
    # print(max(img.flatten()),max(reverted_img.flatten()),min(img.flatten()),min(reverted_img.flatten()))
    
    #apply median filter
    median_filtered = ndimage.median_filter(reverted_img, size=9)
    
    #apply frangi filter - fails so far
    # frangi_img = frangi(median_filtered,(0.1,0.5),0.1,black_ridges=False)
    
    #integrate into dicom
    ds.PixelData = median_filtered.tobytes()
    # print('output:', reverted_img.shape, np.mean(reverted_img))
    # print(ds.pixel_array[300:306,300:306])

    #save dicom images
    print(dicom_img.split('/'))
    output_dir = output_path + subj_id + '/' + dicom_img.split('/')[-2] + '/'
    print(output_dir)
    
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    pydicom.filewriter.write_file(output_dir + dicom_img.split('/')[-1],ds)
    