import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../preprocessing"))
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
import cvxopt
from cvxopt import matrix
from kitti_util import *
import generate_lidar

def _make_anchor(calib, pseudo, lidar, method):
    if method == 'nearest':
        proj_lidar = calib.project_velo_to_image(lidar)  # (N, 2)
        proj_pseudo = calib.project_velo_to_image(pseudo)  # (M, 2)
        tree = KDTree(proj_pseudo)
        _, nn_indices = tree.query(proj_lidar, 1)
        anchor_depth = pseudo[nn_indices[:,0]]

        proj_anchor = np.hstack([
            proj_lidar,
            anchor_depth[:,None]])
        anchor = calib.project_image_to_velo(proj_anchor) # (N, 3)
        return anchor
    else:
        raise NotImplementedError()
    
def gdc(calib, pseudo, max_high, k, lidar, method='nearest'):
    """build KNN Graph and return shifted pseudo lidar.
    Parameters:
        calib: Calilbration
        pseudo: (M, 3) array_like, XYZ
            the pseudo lidar data points
        max_high: float
            the parameter for generating lidar
        k: int
            the number of nearest neighbor
        lidar: (N, 3) array_like, XYZ
            the "ground truth" lidar data points
        method: 'nearest' or 'bilinear'
            the method to project 3d lidar points onto pixel locations
    Returns:
        cloud: (N+M, 3) array_like
    """
    #rows, cols = depth.shape
    #proj_lidar = calib.project_velo_to_image(lidar)
    #proj_lidar_r = proj_lidar[:,0]
    #proj_lidar_c = proj_lidar[:,1]
    #if method == 'nearest':
    #    r = np.floor(proj_lidar_r + 0.5)
    #    c = np.floor(proj_lidar_c + 0.5)
    #    anchor_depth = depth[r, c]
    #elif method == 'bilinear':
    #    raise NotImplementedError()
    #else:
    #    raise NotImplementedError()
    ## TODO: need to ignore lidar points outside image / behind the car (X <= 0 ?)
    

    anchor = _make_anchor(calib, pseudo, lidar, method)
    N = len(anchor)
    
    pseudo = generate_lidar.project_depth_to_points(calib, depth, max_high) # (M, 3)
    M = len(pseudo)

    data = np.concatenate([anchor, pseudo]) # (N+M, 3)
    NM = len(data)

    ## calc the weight of knn-graph (N+M, N+M)
    tree = KDTree(data)
    _, nn_indices = tree.query(data, k+1)
    nn_indices = nn_indices[:, 1:] # exclude myself, (N+M, k)

    Wg_data = np.empty((N, k))
    Wg_ind = []
    Wg_indp = np.empty(N+1, dtype=np.int)
    Wg_indp[0] = 0

    Wz_data = np.empty((M, k))
    Wz_ind = []
    Wz_indp = np.empty(M+1, dtype=np.int)
    Wz_indp[0] = 0
    for i in range(NM):
        nn = data[nn_indices[i]].T # (3, k)
        target = data[i] # (3,)
        one = np.ones((1, k))
        one_target = np.ones(1)
        
        P = matrix(np.eye(k))
        q = matrix(np.zeros(k))
        G = matrix(np.concatenate([nn, -nn, one, -one]))
        h = matrix(np.concatenate([target, -target, one_target, one_target]))
        sol = cvxopt.solvers.qp(P,q,G,h)
        inG = nn_indices[i] < N
        inZ = nn_indices[i] >= N
        Wg_data[i] = sol.x[inG]
        Wg_ind.append(nn_indices[inG])
        Wg_indp[i+1] = Wg_indp[i] + inG.sum()

        Wz_data[i] = sol.x[inZ]
        Wz_ind.append(nn_indices[inZ])
        Wz_indp[i+1] = Wz_indp[i] + inZ.sum()

    Wg = sparse.csr_matrix((Wg_data, Wg_ind, Wg_indp), shape=(NM,N))
    Wz = sparse.csr_matrix((Wz_data, Wz_ind, Wz_indp), shape=(NM,M))
    
    ## retrieve ##
    G = sparse.csc_matrix((lidar.T.reshape(-1),
                           np.tile(np.arange(N), 3),
                           np.arange(4) * N), shape=(N, 3))
    GZ = sparse.csc_matrix(np.concatenate([lidar, pseudo])) # (N+M, 3)

    Zretrieve = np.empty((3, M))
    for i in range(3):
        Zretrieve[i] = sparse.linalg.lsqr(Wz,
                                          GZ.getcol(i) - Wg*G.getcol(i))

    cloud = np.concatenate([lidar, Zretrieve.T])
    return cloud

if __name__ == '__main__':
    import argparse
    import os
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_dir', type=str)
    parser.add_argument('--pseudo_dir', type=str)
    parser.add_argument('--lidar_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='./shifted_valodyne')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--nn', type=int, default=10)
    args = parser.parse_args()
    
    assert os.path.isdir(args.calib_dir)
    assert os.path.isdir(args.pseudo_dir)
    assert os.path.isdir(args.lidar_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    pseudo_fns = [x for x in os.listdir(args.pseudo_dir) if x[-3:] == 'bin']
    pseudo_fns = sorted(pseudo_fns)

    for fn in tqdm(pseudo_fns):
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        lidar_file = '{}/{}.bin'.format(args.lidar_dir, predix)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        pseudo_file = '{}/{}.bin'.format(args.lidar_dir, predix)
        pseudo = np.fromfile(pseudo_file, dtype=np.float32).reshape(-1, 4)
        
        shifted = gdc(calib, pseudo[:,:3], args.max_high, args.nn, lidar[:,:3])
        shifted.astype(np.float32).tofile('{}/{}.bin'.format(args.save_dir, predix))



