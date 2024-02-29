import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# # from .door import Door
# from Graphormer_DRGCN_01.models.position import GraphAttnBias


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        # self.door = Door(1024, 512, dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        # return self.door(x, self.dropout(sublayer(self.norm(x))))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        # init_parameters
    #     self.apply(self.init_parameters)
    #
    # def init_parameters(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.constant_(module.bias, 0)

    def forward(self, query, key, value):
        """Implements Figure 2"""
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(-1, self.h, self.d_k).transpose(0, 1)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     nn.init.xavier_uniform_(self.w_1.weight, gain=nn.init.calculate_gain('relu'))
    #     nn.init.constant_(self.w_1.bias, 0)
    #     nn.init.xavier_uniform_(self.w_2.weight, gain=nn.init.calculate_gain('relu'))
    #     nn.init.constant_(self.w_2.bias, 0)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def set_encoder_model(feature_dim, head_num, model_dim, dropout, layer_num) -> nn.Module:
    mod = nn.Sequential(
        nn.Linear(feature_dim, model_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        Encoder(EncoderLayer(model_dim, MultiHeadedAttention(head_num, model_dim, dropout),
                             PositionwiseFeedForward(model_dim, model_dim * 4, dropout), dropout), layer_num),
    )
    init_parameters(mod)
    return mod


def set_decoder_model(feature_dim, head_num, model_dim, dropout, layer_num) -> nn.Module:
    mod = nn.Sequential(
        Encoder(EncoderLayer(model_dim, MultiHeadedAttention(head_num, model_dim, dropout),
                             PositionwiseFeedForward(model_dim, model_dim * 4, dropout), dropout), layer_num),
        nn.Linear(model_dim, feature_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
    init_parameters(mod)
    return mod


def init_parameters(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)
