import torch
import torch.nn as nn


class CNN(nn.Module):
    """Convolutional neural network."""

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = x.view(x.shape[0], -1)
        return out


def build_cnn_data(features, x, y):
    temp = torch.cat((features[x.tolist()], features[y.tolist()]), 1)
    temp = temp.view(x.shape[0], 1, 2, features.shape[0])
    return temp
