#!/usr/bin/env python3
# coding: utf-8

import torch
from torch.utils.data import DataLoader

from loss import chamfer_loss
from dataloader import PointCloudDataset
from model import CPDNet

if __name__ == '__main__':

    if torch.cuda.is_available():
        print('GPU is available')
        device = torch.device('cuda')
    else:
        print('GPU is not available')
        device = torch.device('cpu')

    epochs = 10000
    log_interval = 100

    train_data = PointCloudDataset('fish', 20000)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

    model = CPDNet().to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.999)

    model.train()
    optimiser.step()

    for epoch in range(epochs):
        lr_scheduler.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'cpd_net.pth')

        for batch_id, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimiser.zero_grad()
            flow_field = model(x, y)
            y_pred = x + flow_field
            loss = chamfer_loss(y_pred, y)

            loss.backward()
            optimiser.step()

            if batch_id % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch + 1, batch_id * len(x), len(train_loader.dataset),
                            100. * batch_id / len(train_loader), loss.item()))
