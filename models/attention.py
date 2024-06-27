import torch
import torch.nn as nn
from torch.nn import init
from .cnn import CNN


class Attention(nn.Module):
    """Implements perspective-level attention mechanism"""

    def __init__(self, in_ft_encoder, in_ft_gcn, out_ft, dropout):
        super(Attention, self).__init__()
        self.cnn = CNN(2, 2, 1, 6)
        self.fc_H = nn.Sequential(
            nn.Linear(3968, 1000, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(1000, 100, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        init.xavier_uniform_(self.fc_H[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
        init.zeros_(self.fc_H[0].bias)
        init.xavier_uniform_(self.fc_H[3].weight, gain=nn.init.calculate_gain('leaky_relu'))
        init.zeros_(self.fc_H[3].bias)

        self.linear_gcn = nn.Sequential(
            nn.Linear(in_ft_gcn, out_ft, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        init.xavier_uniform_(self.linear_gcn[0].weight, gain=nn.init.calculate_gain('tanh'))
        init.zeros_(self.linear_gcn[0].bias)
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_ft_encoder, out_ft, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        init.xavier_uniform_(self.linear_encoder[0].weight, gain=nn.init.calculate_gain('tanh'))
        init.zeros_(self.linear_encoder[0].bias)

        # Randomly initialize a feature matrix of lncRNA and disease,
        # used to calculate the unique measure for each pair of lncRNA-disease.
        self.H = nn.Embedding(645, 1140, padding_idx=0)

        init.xavier_uniform_(self.H.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ld_gcn, ld_encoder, x, y):
        """Follow Figure 4 for connections."""

        feature_gcn = ld_gcn
        feature_encoder = ld_encoder

        # calculate the unique measure for each pair of lncRNA-disease.
        h_ld = torch.cat((self.H(x.cuda()), self.H((y + 240).cuda())), dim=1)
        h_ld = h_ld.view(x.shape[0], 1, 2, 1140)
        h_ld = self.cnn(h_ld)
        h_ld = self.fc_H(h_ld)
        h_ld = h_ld.T

        # Calculate the feature weight of the enhanced perspective using GCNDR
        ld_gcn = self.linear_gcn(ld_gcn)
        re = torch.matmul(ld_gcn, h_ld)
        diag = re.diagonal()
        attention_gcn = diag.view(x.shape[0], 1)

        # Calculate the feature weight of the enhanced perspective using mask
        ld_encoder = self.linear_encoder(ld_encoder)
        re = torch.matmul(ld_encoder, h_ld)
        diag = re.diagonal()
        attention_encoder = diag.view(x.shape[0], 1)

        # Weight normalization and feature Fusion
        att = torch.cat((attention_gcn, attention_encoder), dim=1)
        att = torch.softmax(att, dim=1)
        ld_gcn = feature_gcn * att[:, 0].reshape(x.shape[0], 1)
        ld_encoder = feature_encoder * att[:, 1].reshape(x.shape[0], 1)
        return ld_gcn, ld_encoder
