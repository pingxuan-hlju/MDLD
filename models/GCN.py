import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init


class GCN_conv(nn.Module):
    """Single GCN layer."""

    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = torch.mm(x, self.weight)
        x = torch.spmm(G, x)
        if self.bias is not None:
            x = x + self.bias
        return x


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.gcn_layer1 = GCN_conv(512, 512)
        self.gcn_layer2 = GCN_conv(512, 512)
        self.feat_fc = nn.Linear(1140, 512)

    def forward(self, feat):
        deg = torch.diag((torch.sum(feat > 0, dim=1)) ** (-1 / 2))
        adj_hat = deg @ feat @ deg
        re = feat
        feat = self.feat_fc(feat)
        feat_conv1 = self.gcn_layer1(feat, adj_hat)
        feat_conv1 = feat_conv1 + feat
        feat_conv2 = self.gcn_layer2(feat_conv1, adj_hat)
        feat_conv2 = feat_conv2 + feat_conv1
        feat_conv = torch.cat((feat_conv2, re), dim=1)

        return feat_conv


class GCNDR(nn.Module):
    """Implements GCN with dynamic residuals"""

    def __init__(
            self,
            in_feats,
            n_hidden,
            n_layers,
            dropout,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = in_feats
            out_hidden = n_hidden

            self.convs.append(
                GCN_conv(
                    in_hidden,
                    out_hidden,
                )
            )

            self.norms.append(nn.BatchNorm1d(n_hidden))

        self.alpha_fc = nn.Linear(n_hidden, n_hidden)
        # self.rnn = nn.RNN(n_hidden, 1, 2)
        # self.lstm = nn.LSTM(n_hidden, 1, 2, dropout=0.1)
        self.gru = nn.GRU(n_hidden, 1, 2, dropout=dropout)
        self.feat_fc = nn.Linear(1140, 512)

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

    def init_variables(self, number):
        alpha_hidden = torch.FloatTensor(2, number, 1).cuda()
        bound = 1 / math.sqrt(self.n_hidden)
        return nn.init.uniform_(alpha_hidden, -bound, bound)

    def forward(self, feat):
        deg = torch.diag((torch.sum(feat > 0, dim=1)) ** (-1 / 2))
        adj_hat = deg @ feat @ deg
        h_hidden = self.init_variables(feat.shape[0])
        # c_hidden = self.init_variables(feat.shape[0])
        _hidden = []
        h = self.feat_fc(feat)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        _hidden.append(h)
        for i in range(0, self.n_layers):
            # GCN encoding
            h = self.convs[i](h, adj_hat)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Mean L2 regularization
            h1 = F.normalize(_hidden[i], p=2, dim=1)
            hl = F.normalize(h, p=2, dim=1)
            # Calculate the residual weights
            h_mul = torch.mul(h1, hl).unsqueeze(0)
            h_mul_fc = self.alpha_fc(h_mul)
            # Evolution of residual weights
            alpha_out, (h_hidden) = self.gru(h_mul_fc, h_hidden)
            alpha_out = torch.abs(alpha_out)
            alpha_evo = alpha_out.squeeze(0)
            # Fusion of feature representations from adjacent layers
            h = (1 - alpha_evo) * h + alpha_evo * _hidden[i]
            _hidden.append(h)
        h = torch.cat((h, feat), dim=1)
        return h
