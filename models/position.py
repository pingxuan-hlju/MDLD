import torch
import torch.nn as nn
import time
from Graphormer_DRGCN.utils.load_data import create_feature_matrix, load_data, split_dataset, MyDataset
from torch.utils.data import DataLoader


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
            self, num_degree, hidden_dim
    ):
        super(GraphNodeFeature, self).__init__()

        self.degree_encoder = nn.Embedding(num_degree, hidden_dim, padding_idx=0)

    def forward(self, x, feat):
        degree = x.sum(dim=0)
        degree = torch.ceil(degree)
        degree = degree.type(torch.long)
        node_feature = (
                feat
                + self.degree_encoder(degree)
        )
        return node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
            self,
            num_heads,
            num_spatial,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads

        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

    def forward(self, x):
        x = (x > 0.5).float()
        x.fill_diagonal_(0)
        y = x
        spatial_pos = x
        for dis in range(2, 4):
            y = y @ x
            y = torch.where(y != 0, dis, y)
            spatial_pos = torch.where(spatial_pos == 0, y, spatial_pos)
        spatial_pos = torch.where(spatial_pos == 0, 6, spatial_pos)
        spatial_pos.fill_diagonal_(0)
        # [n_node, n_node, n_head] -> [n_head, n_node, n_node]
        spatial_pos = spatial_pos.type(torch.long)
        print(spatial_pos)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
        # .permute(2, 0, 1)

        return spatial_pos_bias
