import torch
import torch.nn as nn


class Door(nn.Module):
    def __init__(self, in_ft, out_ft, dropout):
        super(Door, self).__init__()
        self.linear_door = nn.Sequential(
            nn.Linear(in_ft, out_ft, bias=True),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_gcn, features):
        feat = torch.cat((feat_gcn, features), dim=1)
        z = self.linear_door(feat)
        feat_door = torch.mul(feat_gcn, z) + torch.mul(features, (1 - z))

        return feat_door
