import torch
from torch import nn
import torch.nn.functional as F

class CPDNet(nn.Module):

    def __init__(self,
                 dims: int=3,
                 mlp_in_layers: list=[16, 64, 128, 256, 512],
                 mlp_out_layers: list=[256, 128]):
        super().__init__()

        self.mlp_in_depth = len(mlp_in_layers)
        self.mlp_out_depth = len(mlp_out_layers)

        mlp_in_layers_dims = [dims] + mlp_in_layers
        mlp_in_chs = [(mlp_in_layers_dims[i-1], mlp_in_layers_dims[i])
                        for i in range(1, len(mlp_in_layers_dims))]

        mlp_out_layers_dims = [2 * mlp_in_layers[-1] + dims] + mlp_out_layers
        mlp_out_chs = [(mlp_out_layers_dims[i-1], mlp_out_layers_dims[i])
                        for i in range(1, len(mlp_out_layers_dims))]

        self.src_mlp = nn.ModuleList([nn.Conv1d(in_ch, out_ch, 1)
                                       for (in_ch, out_ch) in mlp_in_chs])
        self.src_bn = nn.ModuleList([nn.BatchNorm1d(ch) for ch in mlp_in_layers])

        self.tgt_mlp = nn.ModuleList([nn.Conv1d(in_ch, out_ch, 1)
                                       for (in_ch, out_ch) in mlp_in_chs])
        self.tgt_bn = nn.ModuleList([nn.BatchNorm1d(ch) for ch in mlp_in_layers])

        self.out_mlp = nn.ModuleList([nn.Conv1d(in_ch, out_ch, 1)
                                       for (in_ch, out_ch) in mlp_out_chs])
        self.out_bn = nn.ModuleList([nn.BatchNorm1d(ch) for ch in mlp_out_layers])

        self.final_layer = nn.Conv1d(128, dims, 1)

    def forward(self, src, tgt):
        x, y = src, tgt
        for level in range(self.mlp_in_depth):
            x = self.src_mlp[level](x)
            x = self.src_bn[level](x)
            x = F.relu(x)
        x = torch.max(x, dim=-1)[0]

        for level in range(self.mlp_in_depth):
            y = self.tgt_mlp[level](y)
            y = self.tgt_bn[level](y)
            y = F.relu(y)
        y = torch.max(y, dim=-1)[0]

        xy = torch.cat((x, y), dim=1).unsqueeze(-1).repeat(1, 1, src.shape[-1])
        c = torch.cat((src, xy), dim=1)

        for level in range(self.mlp_out_depth):
            c = self.out_mlp[level](c)
            c = self.out_bn[level](c)

        return self.final_layer(c)

# if __name__ == '__main__':
#     net = CPDNet()
#     X = torch.randn(3, 100).unsqueeze(0)
#     Y = torch.randn(3, 100).unsqueeze(0)
#
#     print(net(X, Y).shape)
