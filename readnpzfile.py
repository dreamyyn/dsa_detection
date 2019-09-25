import numpy as np

load_path = '../data/DSA_patch/train/30001_post1_0.npz'

npzfile = np.load(load_path)
for key in npzfile.keys():
    print(key, npzfile[key])
# print(npzfile['sample'].shape)