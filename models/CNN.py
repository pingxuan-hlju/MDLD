import torch
import torch.nn as nn
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, conv_w, conv_h, pool_w, pool_h):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(conv_w, conv_h), padding=1, ),  # 卷积盒（2，2）
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(pool_w, pool_h))  # 池化盒（1，6）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(conv_w, conv_h), padding=1, ),  # 卷积盒（2，2）
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(pool_w, pool_h)),  # 池化盒（1，6）
        )

    #     self.initialize_weights()
    #
    # def initialize_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             init.xavier_uniform_(module.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
    #             if module.bias is not None:
    #                 init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = x.view(x.shape[0], -1)
        return out


def build_cnn_data(features, x, y):
    temp = torch.cat((features[x.tolist()], features[y.tolist()]), 1)
    temp = temp.view(x.shape[0], 1, 2, features.shape[0])
    return temp
