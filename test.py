import torch
from torch.utils.data import DataLoader

from loss import chamfer_loss
from dataloader import PointCloudDataset
from model import CPDNet

import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':


    train_data = PointCloudDataset('bunny.txt', 20000)
    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    model = CPDNet()
    model.load_state_dict(torch.load('cpd_net.pth'))
    model.eval()

    max_points = 1000

    x = train_data.normalise(np.loadtxt('FM.txt'))
    y = train_data.normalise(np.loadtxt('EM.txt'))

    x_stride = x.shape[0] // max_points
    y_stride = y.shape[0] // max_points

    np.random.shuffle(x)
    np.random.shuffle(y)

    x = torch.FloatTensor(np.rollaxis(x[::x_stride], axis=-1)[:,:max_points])
    y = torch.FloatTensor(np.rollaxis(y[::y_stride], axis=-1)[:,:max_points])

    with torch.no_grad():
        (x, y) = train_data[0]
        print(x.shape, y.shape)
        flow_field = model(x.unsqueeze(0), y.unsqueeze(0))
        y_pred = x + flow_field
        print(y_pred.shape)
    x_np, y_np, y_pred_np = x.squeeze().numpy(), y.squeeze().numpy(), y_pred.squeeze().numpy()

    visualise(np.rollaxis(x_np, axis=-1),
              np.rollaxis(y_np, axis=-1),
              np.rollaxis(y_pred_np, axis=-1),
              'src',
              'tgt',
              'src_wrpd',
              labels=True)
