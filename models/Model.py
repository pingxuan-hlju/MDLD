import torch
import torch.nn as nn
from .GCN import GCN, DRGCN
from .CNN import CNN
from .attention import Attention


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.gcn = GCN()
        self.dr_gcn = DRGCN(512, 512, 2, 0.4)
        self.cnn_encoder = CNN(2, 2, 1, 3)
        self.cnn_gcn = CNN(2, 2, 1, 3)
        self.att = Attention(1000, 1000, 100, dropout)

        self.fc_gcn = nn.Sequential(
            nn.Linear(23552, 1000, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(23552, 1000, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2000, 1000, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(1000, 100, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(100, 2, bias=True),
        )
        # 初始化模型参数
        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, feat, features, x, y):
        # # # dr_gcn
        feat_gcn = self.dr_gcn(features)
        pair_ld = torch.cat((feat_gcn[x], feat_gcn[y + 240]), 1)
        pair_ld = pair_ld.view(x.shape[0], 1, 2, feat_gcn.shape[1])
        ld = self.cnn_gcn(pair_ld)
        ld = self.fc_gcn(ld)
        # # pre = self.fc(ld)

        # mask_trans
        feat = torch.cat((feat, features), dim=1)
        pair_l_en = feat[x]
        pair_d_en = feat[y + 240]
        pair_ld_en = torch.cat((pair_l_en, pair_d_en), 1)
        pair_ld_en = pair_ld_en.view(x.shape[0], 1, 2, feat.shape[1])
        ld_en = self.cnn_encoder(pair_ld_en)
        ld_en = self.fc_encoder(ld_en)
        # pre = self.fc(ld_en)

        # Attention
        ld, ld_en = self.att(ld, ld_en, x, y)

        final_ld = torch.cat((ld_en, ld), 1)
        pre = self.fc(final_ld)

        return pre
