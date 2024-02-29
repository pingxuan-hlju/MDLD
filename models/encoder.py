from functools import partial
import torch
import torch.nn as nn
from .loss_func import sce_loss
from .transformer import set_encoder_model, set_decoder_model
from .position import GraphNodeFeature
import math


class PreModel(nn.Module):
    def __init__(self,
                 args,
                 ):
        super(PreModel, self).__init__()
        self.feature_dim = args.feature_dim
        self.head_num = args.head_num
        self.model_dim = args.model_dim
        self.dropout = args.dropout
        self.layer_num = args.layer_num
        self.loss_fn = args.loss_fn
        self.dec_in_dim = args.dec_in_dim
        self.feat_mask_rate = args.feat_mask_rate
        self.enc_dec_input_dim = args.enc_dec_input_dim

        self.encoder = set_encoder_model(
            feature_dim=self.feature_dim,
            head_num=self.head_num,
            model_dim=self.model_dim,
            dropout=self.dropout,
            layer_num=self.layer_num,
        )
        self.decoder = set_decoder_model(
            feature_dim=self.feature_dim,
            head_num=self.head_num,
            model_dim=self.model_dim,
            dropout=self.dropout,
            layer_num=self.layer_num,
        )
        self.center = GraphNodeFeature(self.feature_dim, self.model_dim)
        self.alpha_l = args.alpha_l
        self.attr_restoration_loss = self.setup_loss_fn(self.loss_fn, self.alpha_l)
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self._replace_rate = args.replace_rate
        self._leave_unchanged = args.leave_unchanged
        assert self._replace_rate + self._leave_unchanged < 1, "Replace rate + leave_unchanged must be smaller than 1"

    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, features, **kwargs):
        loss, feat_recon, enc_out, mask_nodes = self.mask_attr_restoration(features, kwargs.get("epoch", None))
        return loss, feat_recon, enc_out, mask_nodes

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate: list[float] = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def encoding_mask_noise(self, x, mask_rate):
        num_nodes = x.shape[0] - 495
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self._leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, epoch):
        cur_feat_mask_rate = self.get_mask_rate(self.feat_mask_rate, epoch=epoch)
        # cur_feat_mask_rate = 0.0
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_feat_mask_rate)
        enc_out = self.encoder(use_x)
        enc_out_mapped = self.encoder_to_decoder(enc_out)
        enc_out_mapped[mask_nodes] = 0  # TODO: learnable? remove?
        enc_out_mapped[mask_nodes] += self.dec_mask_token
        enc_out_mapped = self.center(feat, enc_out_mapped)
        feat_recon = self.decoder(enc_out_mapped)

        # x_init = feat[mask_nodes]
        # x_rec = feat_recon[mask_nodes]
        x_init = feat
        x_rec = feat_recon
        loss = self.attr_restoration_loss(x_rec, x_init)

        return loss, feat_recon, enc_out, mask_nodes

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def get_embeds(self, feats):
        rep = self.encoder(feats)
        return rep.detach()
