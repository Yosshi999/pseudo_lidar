import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../preprocessing"))
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
import cvxopt
from cvxopt import matrix
from kitti_util import *
from generate_lidar import project_depth_to_points

def build_knn_graph(calib, k, lidar, depth, max_high=1, method='nearest'):
    """build KNN Graph and return its weight.
    Parameters:
        calib: Calilbration
        k: int
            the number of nearest neighbor
        lidar: (N, 3) array_like, XYZ
            the "ground truth" lidar data points
        depth: (H, W) array_like
            the pseudo lidar depth map
        max_high: float
            the parameter for generating lidar
        method: 'nearest' or 'bilinear'
            the method to project 3d lidar points onto pixel locations
    Returns:
        weight: (
    """
    rows, cols = depth.shape
    proj_lidar = calib.project_velo_to_image(lidar)
    proj_lidar_r = proj_lidar[:,0]
    proj_lidar_c = proj_lidar[:,1]
    if method == 'nearest':
        r = np.floor(proj_lidar_r + 0.5)
        c = np.floor(proj_lidar_c + 0.5)
        anchor_depth = depth[r, c]
    elif method == 'bilinear':
        raise NotImplementedError()
    else:
        raise NotImplementedError()


    proj_anchor = np.hstack([
        proj_lidar_c[:,None],
        proj_lidar_r[:,None],
        anchor_depth[:,None]])
    anchor = calib.project_image_to_velo(proj_anchor) # (N, 3)
    N = len(anchor)
    
    pseudo = project_depth_to_points(calib, depth, max_high) # (M, 3)
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
    return Zretrieve.T

