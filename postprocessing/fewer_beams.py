"""simulate 4-beam lidar"""

import numpy as np
import argparse
import os
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='./4beam_lidar')
    args = parser.parse_args()

    assert os.path.isdir(args.lidar_dir)
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
        fewer = lidar[mask]
        fewer.tofile('{}/{}.bin'.format(args.save_dir, predix))



