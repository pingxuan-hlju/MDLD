import torch
import torch.nn as nn


class GraphNodeFeature(nn.Module):
    """
    Compute central position information for each node in the graph.
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
