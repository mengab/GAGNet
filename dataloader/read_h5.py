import numpy as np
import cv2
import torch.multiprocessing as mp
mp.set_start_method('spawn')
import h5py

f = h5py.File('../train_QP32.h5','r')

for key in f.keys():
    print(f[key].name,f[key].shape)