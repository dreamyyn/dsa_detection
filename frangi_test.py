import pydicom
from skimage.filters import frangi, hessian
import numpy as np
import itk
import matplotlib.pyplot as plt
import pickle
from skimage.transform import rescale

##visualize dicom
# dicom_img = '/Volumes/My Passport/StanfordRSL/dsa_preprocess_data/dsa_AP/30150_size9/RICA_2/IM-0001-0306.dcm'
# ds = pydicom.dcmread(dicom_img)
# image = ds.pixel_array
# post_process = frangi(image)
# ds.PixelData = post_process.tobytes()
# print(post_process[:20,:20],ds.pixel_array[:20,:20])


#visualize npz
data = np.load('../data/DSA_patch/train/30131_LCCA_5_pos_slice2_0.npz')
image = data['sample'][:,:,3]

# # visualize pickle
# with open('../data/DSA_pkl/' + '30131' + '.pickle', 'rb') as handle:
#     data = pickle.load(handle)
# image_dict = data['image_dict']
# for key in image_dict.keys():
#     print(key)
# image = image_dict[key][:,:,4]
# print(image.shape)

rescaled_image = rescale(image, 1.5, anti_aliasing=True)
print(image.shape, rescaled_image.shape)

fig, ax = plt.subplots(ncols=4)

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(frangi(image), cmap=plt.cm.gray)
ax[1].set_title('Frangi filter result')

ax[2].imshow(hessian(image,(1,5),1), cmap=plt.cm.gray)
ax[2].set_title('Hybrid Hessian filter result')

ax[3].imshow(rescaled_image[:128,:128], cmap=plt.cm.gray)
ax[3].set_title('')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()