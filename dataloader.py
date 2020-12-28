#!/usr/bin/env python3
# coding: utf-8
import os
import torch
import math
from torch.utils.data import Dataset
import numpy as np
from numpy.random import Generator, PCG64

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
        if x.shape[-1] == 3:
            ax.scatter(x[:,0],  x[:,1], x[:,2], color=color_dict[i], label=label)
        elif x.shape[-1] == 2:
            ax.scatter(x[:,0],  x[:,1], 0, color=color_dict[i], label=label)
        ax.legend(loc='upper left', fontsize='x-large')
    plt.show()

class PointCloudDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 data_len: int,
                 rand_seed: int=12345):

        self.data_len = data_len

        if data_path == 'fish':
            sc  = [-0.9154191606171814, -0.16535078775508855, -0.890508968073163, -0.10212842773109136, -0.8572953780144712, -0.10212842773109136, -0.8240817879557792, -0.1495451977490876, -0.8739021730438176, -0.19696196776709046, -0.923722558131855, -0.16535078775508855, -1.0648803158812945, -0.2917955078030896, -1.02336332830793, -0.16535078775508855, -0.9818463407345652, -0.05471165771308847, -0.9569361481905461, 0.024316292316909686, -0.9154191606171814, 0.11914983235291547, -0.8822055705584902, 0.18237219237691266, -0.8074749929264337, 0.26140014240691745, -0.7576546078383953, 0.34042809243691563, -0.674620632691666, 0.4668728124849167, -0.5832832600302639, 0.5459007625149215, -0.5168560799128809, 0.6091231225389186, -0.4006085147074595, 0.7039566625749244, -0.3341813345900757, 0.8145957926169245, -0.26775415447269185, 1.1623187727489324, -0.201326974355308, 1.4626249828629307, -0.15150658926727137, 1.7629311929769422, -0.1432031917525986, 1.8893759130249435, -0.11829299920858027, 2.1106541731089434, -0.10168620417923308, 1.9684038630549416, -0.0933828066645603, 1.7471256029709414, -0.0933828066645603, 1.5416529328929356, -0.08507940914988753, 1.3045690828029344, -0.0933828066645603, 1.1149020027309295, -0.07677601163521475, 0.9094293326529237, -0.07677601163521475, 0.7197622525809254, -0.043562421576523666, 0.5617063525209225, -0.03525902406184923, 0.48267840249091765, 0.04777495108487849, 0.40365045246091946, 0.11420213120226233, 0.34042809243691563, 0.16402251629030062, 0.27720573241291846, 0.2221462988930117, 0.21398337238891457, 0.29687687652506667, 0.1033442423469145, 0.36330405664245047, -0.007294887695085575, 0.44633803178917986, -0.08632283772509039, 0.504461814391891, -0.05471165771308847, 0.5708889945092748, 0.04012188232291065, 0.6041025845679658, 0.18237219237691266, 0.6705297646853497, 0.3878448624549185, 0.7286535472880591, 0.6249287125449197, 0.8199909199494629, 0.6723454825629225, 0.9030248950961923, 0.7671790225989217, 0.8864181000668467, 0.6249287125449197, 0.8698113050375013, 0.4668728124849167, 0.8615079075228268, 0.26140014240691745, 0.8365977149788085, 0.1033442423469145, 0.8449011124934812, -0.1337396077430933, 0.8449011124934812, -0.30760109780909056, 0.8449011124934812, -0.5288793578930974, 0.8781147025521724, -0.7343520279710966, 0.9279350876402106, -0.8924079280310996, 0.9860588702429217, -1.0030470580731063, 1.03587925533096, -1.1294917781211073, 1.044182652845631, -1.2085197281511055, 0.9694520752135745, -1.1452973681271084, 0.8864181000668467, -1.0504638280911025, 0.7950807274054447, -0.9714358780611043, 0.6871365597146952, -0.9240191080431015, 0.5957991870532932, -0.8291855680071023, 0.537675404450582, -0.7501576179770976, 0.47124822433319985, -0.6395184879350975, 0.4131244417304888, -0.5762961279111003, 0.2885734790103939, -0.5288793578930974, 0.18893270883431895, -0.46565699786909354, 0.08929193865824402, -0.46565699786909354, -0.03525902406184923, -0.49726817788109545, -0.08507940914988753, -0.49726817788109545, -0.0020454340031581387, -0.5762961279111003, 0.0975953361729168, -0.6553240779410984, 0.1723259138049734, -0.6869352579531003, 0.26366328646637555, -0.7343520279710966, 0.21384290137833725, -0.7343520279710966, 0.03947155357020572, -0.7343520279710966, -0.15150658926727137, -0.7343520279710966, -0.26775415447269185, -0.7185464379651023, -0.4006085147074595, -0.6711296679470994, -0.46703569482484336, -0.6079073079230956, -0.5583730674862455, -0.5288793578930974, -0.5998900550596102, -0.5130737678870964, -0.7742614028677418, -0.46565699786909354, -0.890508968073163, -0.4814625878750945, -0.9818463407345652, -0.43404581785709156, -1.0399701233372756, -0.3708234578330944, -1.0565769183666218, -0.2917955078030896, -0.05048191950541625, -0.7580604129800946, 0.09967118555158623, -0.7580604129800946]
            self.source = self.normalise(np.asarray(sc).reshape(-1,2))

        elif os.path.splitext(data_path)[1] == 'txt':
            self.source = self.normalise(np.loadtxt(data_path))

        # Create training set from reproducible random deformations
        rg = Generator(PCG64(rand_seed))
        self.deformations = np.asarray([[rg.uniform(-0.3, 0.3, (1, self.source.shape[-1]))
                                for __ in range(len(self.source))]
                                    for __ in range(data_len)])

        assert self.source.ndim == 2, 'Point cloud data must have dimension 2'
        assert self.source.shape[-1] == 2 or self.source.shape[-1] == 3, 'Point cloud data must be 2D or 3D'

    # Randomly generate source and target data pairs
    def __getitem__(self, idx):

        x = self.source
        x = torch.FloatTensor(np.rollaxis(x, axis=-1))

        y = self.normalise(self.apply_tps_deformation(self.source, idx))
        y = torch.FloatTensor(np.rollaxis(y, axis=-1))

        return x, y


    def __len__(self):
        return self.data_len

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        assert np.min(arr) != np.max(arr)

        def norm(arr):
            return 2 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) - 1

        x_arr = arr[:,0]
        y_arr = arr[:,1]
        if arr.shape[-1] == 3:
            z_arr = arr[:,2]

        x_norm = norm(x_arr)
        y_norm = norm(y_arr)
        if arr.shape[-1] == 3:
            z_norm = norm(z_arr)

        arr[:,0] = x_norm
        arr[:,1] = y_norm
        if arr.shape[-1] == 3:
            arr[:,2] = z_norm
        return arr

    def apply_tps_deformation(self, arr: np.ndarray, idx: int, num_of_points: int=3) -> np.ndarray:
        # Create grid of control points
        if arr.shape[-1] == 3:
            control_points = self.normalise(np.asarray([[x, y, z] for x in [i for i in range(num_of_points)]
                                                    for y in [i for i in range(num_of_points)]
                                                     for z in [i for i in range(num_of_points)]]))

        elif arr.shape[-1] == 2:
            control_points = self.normalise(np.asarray([[x, y] for x in [i for i in range(num_of_points)]
                                                    for y in [i for i in range(num_of_points)]]))
        # Randomly perturb control points
        target_points = np.stack([np.squeeze(control_points[i] + self.deformations[idx][i])
                            for i in range(control_points.shape[0])])

        trans = TPS(control_points, target_points)

        return trans(arr)
