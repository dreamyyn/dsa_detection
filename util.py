import argparse
from scipy import io as sio
import numpy as np
import os
# import dicom
# import nibabel as nib
import sys
import pdb
import random
from glob import glob

def load_data(data_path):
    # print('loading data {}'.format(data_path))
    data = np.load(data_path)
    label = data['label']
    image = data['sample']

    return image, label
