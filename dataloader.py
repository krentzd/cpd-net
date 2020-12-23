import torch
import math
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt

# Adapted from: https://github.com/tzing/tps-deformation
class TPS:

    def __init__(self,
                control_points: np.ndarray,
                target_points: np.ndarray,
                lmbd: float=0,
                solver: str='exact'):

        self.control_points = control_points
        self.coefficients = self.find_coefficients(control_points,
                                                   target_points,
                                                   lmbd,
                                                   solver)

    def __call__(self,
                 source_points: np.ndarray):

        return self.transform(source_points,
                              self.control_points,
                              self.coefficients)

    def find_coefficients(self,
                          control_points: np.ndarray,
                          target_points: np.ndarray,
                          lmbd: float=0,
                          solver: str='exact') -> np.ndarray:
        p, d = control_points.shape

        # Create matrix
        K = self.pairwise_radial_basis(control_points, control_points)
        P = np.hstack([np.ones((p, 1)), control_points])

        # Add regularisation
        K = K + lmbd * np.identity(p)

        # Target points
        M = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((d + 1, d + 1))])
        ])
        Y = np.vstack([target_points, np.zeros((d + 1, d))])

        solver = solver.lower()
        if solver == 'exact':
            X = np.linalg.solve(M, Y)
        elif solver == 'lstsq':
            X, __, __, __ = np.linalg.lstsq(M, Y, None)

        return X

    def pairwise_radial_basis(self,
                              K: np.ndarray,
                              B: np.ndarray) -> np.ndarray:

        r_mat = self.cdist(K, B)

        pairwise_cond_1 = r_mat >= 1
        pairwise_cond_2 = r_mat < 1
        r_mat_1 = r_mat[pairwise_cond_1]
        r_mat_2 = r_mat[pairwise_cond_2]

        P = np.empty(r_mat.shape)
        P[pairwise_cond_1] = (r_mat_1**2) * np.log(r_mat_1)
        P[pairwise_cond_2] = r_mat_2 * np.log(np.power(r_mat_2, r_mat_2))

        return P

    def cdist(self,
              K: np.ndarray,
              B: np.ndarray) -> np.ndarray:

        K = np.expand_dims(K, 1)
        B = np.expand_dims(B, 0)
        D = K - B

        return np.linalg.norm(D, axis=2)

    def transform(self,
                  source_points: np.ndarray,
                  control_points: np.ndarray,
                  coefficients: np.ndarray) -> np.ndarray:

        n = source_points.shape[0]
        A = self.pairwise_radial_basis(source_points, control_points)
        K = np.hstack([A, np.ones((n, 1)), source_points])

        deformed_points = np.dot(K, coefficients)

        return deformed_points


def get_T(t_x, t_y, t_z, s_x, s_y, s_z, theta_x, theta_y, theta_z):

    T_a = np.array([[1., 0,  0, t_x],
                    [0,  1., 0, t_y],
                    [0,  0,  1., t_z],
                    [0,  0,  0,  1.]])

    T_x = np.array([[1., 0,  0, 0],
                    [0,  math.cos(theta_x), -math.sin(theta_x), 0],
                    [0,  math.sin(theta_x),  math.cos(theta_x), 0],
                    [0,  0,  0,  1.]])

    T_y = np.array([[math.cos(theta_y), 0,  math.sin(theta_y), 0],
                    [0,  1,  0, 0],
                    [-math.sin(theta_y),  0,  math.cos(theta_y), 0],
                    [0,  0,  0,  1.]])

    T_z = np.array([[math.cos(theta_z), math.sin(theta_z),  0, 0],
                    [-math.sin(theta_z),  math.cos(theta_z), 0, 0],
                    [0,  0,  1., 0],
                    [0,  0,  0,  1.]])

    T_s = np.array([[s_x, 0, 0, 0],
                    [0, s_y, 0, 0],
                    [0, 0, s_z, 0],
                    [0, 0, 0, 1.]])

    T = T_a @ T_x @ T_y @ T_z @ T_s

    return T

def get_random_T():
    # Random affine transform
    T = get_T(t_x=0,
              t_y=0,
              t_z=0,
              s_x=1,
              s_y=1,
              s_z=1,
              theta_x=np.random.uniform(-math.pi / 8, math.pi / 8),
              theta_y=np.random.uniform(-math.pi / 8, math.pi / 8),
              theta_z=np.random.uniform(-math.pi / 8, math.pi / 8))
    return T

def visualise(*X, labels=False):

    color_dict = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'cyan', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray', 9: 'olive'}

    if labels:
        labels = X[int(len(X) / 2):]
        X = X[:int(len(X) / 2)]
    else:
        labels = range(len(X))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (x, label) in enumerate(zip(X, labels)):
        ax.scatter(x[:,0],  x[:,1], x[:,2], color=color_dict[i], label=label)
        ax.legend(loc='upper left', fontsize='x-large')
    plt.show()

class PointCloudDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 data_len: int):

        self.data_len = data_len

        self.source = self.normalise(np.loadtxt(data_path))

        assert self.source.ndim == 2, 'Point cloud data must have dimension 2'
        assert self.source.shape[1] == 2 or self.source.shape[1] == 3, 'Point cloud data must be 2D or 3D'

    # Randomly generate source and target data pairs
    def __getitem__(self, idx):

        src_tmp = np.column_stack((self.source, np.ones(self.source.shape[0],)))

        # Generate src
        # src = np.stack([np.matmul(T, src_tmp[i])
        #                     for i in range(self.source.shape[0])])
        T = get_random_T()
        # Generate tgt
        tgt = np.stack([np.matmul(T, src_tmp[i])
                            for i in range(self.source.shape[0])])
        src = src_tmp

        return torch.FloatTensor(np.rollaxis(src[:,:3], axis=-1)), torch.FloatTensor(np.rollaxis(tgt[:,:3], axis=-1))

        # max_points = 2000
        #
        # x = self.normalise(np.loadtxt('FM.txt'))
        # # y = self.normalise(np.loadtxt('EM.txt'))
        # y = self.normalise(self.apply_tps_deformation(x))
        #
        # x_stride = x.shape[0] // max_points
        # # y_stride = y.shape[0] // max_points
        #
        # np.random.shuffle(x)
        # # np.random.shuffle(y)
        #
        # x = torch.FloatTensor(np.rollaxis(x[::x_stride], axis=-1)[:,:max_points])
        # # y = torch.FloatTensor(np.rollaxis(y[::y_stride], axis=-1)[:,:max_points])
        #
        # y = torch.FloatTensor(np.rollaxis(y[::x_stride], axis=-1)[:,:max_points])
        #
        # return x, y

    def __len__(self):
        return self.data_len

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        assert np.min(arr) != np.max(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def apply_tps_deformation(self, arr: np.ndarray) -> np.ndarray:
        # Create grid of control points
        control_points = np.asarray([[x, y, z] for x in [0.1 * i for i in range(11)]
                                                for y in [0.1 * i for i in range(11)]
                                                 for z in [0.1 * i for i in range(11)]])

        # Randomly perturb control points
        if np.random.random() > 0.5:
            target_points = np.stack([np.squeeze(control_points[i] + np.random.uniform(0, 0.1, (1, 3)))
                                for i in range(control_points.shape[0])])
        else:
            target_points = np.stack([np.squeeze(control_points[i] + np.random.uniform(-0.1, 0, (1, 3)))
                                for i in range(control_points.shape[0])])

        trans = TPS(control_points, target_points)

        return trans(arr)

# if __name__ == '__main__':
#
#     pc_data = PointCloudDataset('bunny.txt', 10)
#
#     src, tgt = pc_data[0]
#
#     # Control points are a grid of spaced points
#     control_points = np.asarray([[x, y, z] for x in [0.1 * i for i in range(11)]
#                                             for y in [0.1 * i for i in range(11)]
#                                              for z in [0.1 * i for i in range(11)]])
#     # Target points are randomly perturbed control points
#     target_points = np.stack([np.squeeze(control_points[i] + np.random.uniform(0, 0.05, (1, 3)))
#                         for i in range(control_points.shape[0])])
#     print(control_points, target_points)
#     print(np.squeeze(src))
#     trans = TPS(control_points, target_points)
#
#     # src_wrpd = trans(np.squeeze(np.rollaxis(src.numpy(), axis=-1)))
#
#     # src_wrpd = pc_data.apply_tps_deformation(np.squeeze(np.rollaxis(src.numpy(), axis=-1)))
#
#     visualise(np.squeeze(np.rollaxis(src.numpy(), axis=-1)), np.squeeze(np.rollaxis(tgt.numpy(), axis=-1)))
#     # visualise(control_points, target_points)
