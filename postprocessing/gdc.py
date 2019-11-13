import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../preprocessing"))
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import open3d as o3d
from kitti_util import *
import generate_lidar

def _make_anchor(calib, pseudo, lidar, method):
    if method == 'nearest':
        print("M=", len(pseudo))
        print("N=", len(lidar))
        proj_lidar = calib.project_velo_to_image(lidar)  # (N, 2)
        proj_pseudo = calib.project_velo_to_image(pseudo)  # (M, 2)

        proj_lidar_ = np.hstack([proj_lidar, np.zeros((len(lidar), 1))])
        proj_pseudo_ = np.hstack([proj_pseudo, np.zeros((len(pseudo), 1))])

        print("anchor start");_t = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(proj_pseudo_)
        tree = o3d.geometry.KDTreeFlann(pcd)
        nn_indices = np.empty(len(proj_lidar), dtype=np.int)
        for i, q in enumerate(proj_lidar_):
            _, nn_idx, _ = tree.search_knn_vector_3d(q, 1)
            nn_indices[i] = nn_idx[0]
        #_, nn_indices = tree.query(proj_lidar, 1)
        print("anchor done", time.time() - _t)

        # anchor_depth = pseudo[nn_indices, 2]
        # proj_anchor = np.hstack([
        #     proj_lidar,
        #     anchor_depth[:,None]])
        # anchor = calib.project_image_to_velo(proj_anchor) # (N, 3)
        anchor = pseudo[nn_indices]
        return anchor
    else:
        raise NotImplementedError()

def drop_outside_image(calib, lidar):
    # only front
    mask = lidar[:,0] > 0

    img = calib.project_velo_to_image(lidar[:,:3])
    mask &= (img[:,0] > 0) & (img[:,0] < 1242) & (img[:,1] > 0) & (img[:,1] < 375)
    fewer = lidar[mask]
    return fewer
    
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
    
    #pseudo = generate_lidar.project_depth_to_points(calib, depth, max_high) # (M, 3)
    #pseudo = drop_outside_image(calib, pseudo)
    M = len(pseudo)

    data = np.concatenate([anchor, pseudo]) # (N+M, 3)
    NM = len(data)

    ## calc the weight of knn-graph (N+M, N+M)
    print("knn start");_t=time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    tree = o3d.geometry.KDTreeFlann(pcd)
    nn_indices = np.empty((NM, k), dtype=np.int)
    for i, q in enumerate(data):
        _, nn_idx, _ = tree.search_knn_vector_3d(q, k+1)
        nn_indices[i] = nn_idx[1:] # exclude myself

    # tree = KDTree(data)
    # _, nn_indices = tree.query(data, k+1)
    # nn_indices = nn_indices[:, 1:] # exclude myself, (N+M, k)
    print("knn done", time.time() - _t)

    Wg_data = []
    Wg_ind = []
    Wg_indp = np.empty(NM+1, dtype=np.int)
    Wg_indp[0] = 0

    Wz_data = []
    Wz_ind = []
    Wz_indp = np.empty(NM+1, dtype=np.int)
    Wz_indp[0] = 0

    print("building knn-graph...");_t=time.time()

    sol = np.empty(k)
    for i in tqdm(range(NM)):
        nn = data[nn_indices[i]].T # (3, k)
        target = data[i] # (3,)
        A = nn[:,1:] - nn[:,0:1]
        An = np.concatenate([A, np.eye(k-1), np.ones((1,k-1))], axis=0)
        b = np.concatenate([target - nn[:,0], np.zeros(k-1), np.ones(1)])
        sol_ = np.linalg.lstsq(An, b)[0]
        sol[0] = 1 - sol_.sum()
        sol[1:] = sol_
        inG = nn_indices[i] < N
        inZ = nn_indices[i] >= N
        Wg_data.append(sol[inG])
        Wg_ind.append(nn_indices[i][inG])
        Wg_indp[i+1] = Wg_indp[i] + inG.sum()

        Wz_data.append(sol[inZ])
        Wz_ind.append(nn_indices[i][inZ] - N)
        Wz_indp[i+1] = Wz_indp[i] + inZ.sum()

    Wz_ind = np.concatenate(Wz_ind)
    Wg_ind = np.concatenate(Wg_ind)
    Wg_data = np.concatenate(Wg_data)
    Wz_data = np.concatenate(Wz_data)

    Wg = sparse.csr_matrix((Wg_data, Wg_ind, Wg_indp), shape=(NM,N))
    Wz = sparse.csr_matrix((Wz_data, Wz_ind, Wz_indp), shape=(NM,M))
    ## save retrieved Z
    #np.concatenate([(Wg * lidar + Wz * pseudo), np.ones((NM,1))], axis=1).astype(np.float32).tofile("reference.bin") 
    Wg.setdiag(-1, k=0)
    Wz.setdiag(-1, k=-N)
    print(time.time() - _t)
    
    ## retrieve ##
    print("retrieve...");_t=time.time()

    Zretrieve = np.empty((3, M))
    for i in range(3):
        progress = tqdm()
        b = Wg.multiply(lidar[:,i]).sum(axis=1)
        b = -np.asarray(b).reshape(-1)
        A = sparse.linalg.LinearOperator((NM, NM), matvec=lambda x: Wz.multiply(x[N:]).sum(axis=1))
        Zretrieve[i] = sparse.linalg.gmres(A, b, x0=data[:,i], maxiter=300, callback=lambda _:progress.update(1))[0][N:]

    print("make cloud")
    cloud = np.concatenate([lidar, Zretrieve.T])
    print(time.time() - _t)
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
        calib = Calibration(calib_file)
        lidar_file = '{}/{}.bin'.format(args.lidar_dir, predix)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        pseudo_file = '{}/{}.bin'.format(args.pseudo_dir, predix)
        pseudo = np.fromfile(pseudo_file, dtype=np.float32).reshape(-1, 4)
        
        shifted = gdc(calib, pseudo[:,:3], args.max_high, args.nn, lidar[:,:3])
        print(shifted.shape)
        shifted = np.concatenate([shifted, np.ones((len(shifted), 1))], axis=1)
        print(shifted.shape)
        shifted.astype(np.float32).tofile('{}/{}.bin'.format(args.save_dir, predix))



