# /usr/bin/env python3

import h5py
import torch
import numpy as np
dataset_folder="/home/olaf/DLC/dataset/"
with h5py.File(dataset_folder+"nyu_depth_v2_labeled.mat",'r') as f:
    print(f['images'], f['depths'])

    path=dataset_folder+"train/"
    for i in range(1200):
        if i%100==0:
            print(i )
        image=f['images'][i,:,:,:]
        depth=f['depths'][i,:,:]

        with h5py.File(path+str(i)+'.h5','w') as h5:
            h5['rgb']=image
            h5['depth']=depth

    path=dataset_folder+"val/"
    for i in range(1200,1500):
        if i%100==0:
            print(i )
        image=f['images'][i,:,:,:]
        depth=f['depths'][i,:,:]

        with h5py.File(path+str(i)+'.h5','w') as h5:
            h5['rgb']=image
            h5['depth']=depth
        

# checkpoint = torch.load("/home/olaf/DLC/CV_LTH_Pre-training/Detection/weight/moco_v2_800ep_pretrain.pth.tar")
# print(checkpoint['arch'])
# print(checkpoint['epoch'])
