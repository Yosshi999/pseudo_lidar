"""simulate 4-beam lidar"""

import numpy as np
import argparse
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../preprocessing"))

from kitti_util import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_dir', type=str)
    parser.add_argument('--calib_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='./4beam_lidar')
    args = parser.parse_args()

    assert os.path.isdir(args.lidar_dir)
    assert os.path.isdir(args.calib_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    fns = sorted(
            [x for x in os.listdir(args.lidar_dir) if x[-3:] == 'bin'])
    for fn in tqdm(fns):
        predix = fn[:-4]
        lidar = np.fromfile('{}/{}.bin'.format(args.lidar_dir, predix), dtype=np.float32).reshape(-1, 4)
        XYZ = np.linalg.norm(lidar[:,:3], axis=1)
        theta = np.arcsin(lidar[:,2]/XYZ) / np.pi * 180
        mask = (
            ((theta >= 0) & (theta < 0.4)) |
            ((theta >= -0.8) & (theta < -0.4)) |
            ((theta >= -1.6) & (theta < -1.2)) |
            ((theta >= -2.4) & (theta < -2.0)))
        
        # only front
        mask &= lidar[:,0] > 0

        # only inside image range
        calib = Calibration('{}/{}.txt'.format(args.calib_dir, predix))
        img = calib.project_velo_to_image(lidar[:,:3])
        mask &= (img[:,0] > 0) & (img[:,0] < 1242) & (img[:,1] > 0) & (img[:,1] < 375)
        
        fewer = lidar[mask]
        fewer.tofile('{}/{}.bin'.format(args.save_dir, predix))



