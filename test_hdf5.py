import h5py
import numpy as np
import cv2

with h5py.File('/home/vision/dualarm_ws/src/dualarm_il/data/1231_2017/common_data.hdf5', 'r') as f:
    # open dataset and listup keys

    demo_len = len(f['data'])
    print("Demo length:", demo_len)

    target_demo = 'demo_10'
    data = f[f'data/{target_demo}/observations']

    for i in range(len(data['joint_L'])):
        image_F = data['image_F'][i]
        image_H = data['image_H'][i]

        cv2.imshow('image_F', image_F)
        cv2.imshow('image_H', image_H)
        # cv2.waitKey(100)
        cv2.waitKey()
    # print("File keys:", list(f['data/demo_0/observations'].keys()))
    # print("Dataset keys:", list(dset.keys()))