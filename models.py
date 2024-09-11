import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class PDN_S(nn.Module):
    def __init__(self, last_layer=384):
        super(PDN_S, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=last_layer, kernel_size=4, padding=0, stride=1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.activation = nn.functional.relu()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        return x


class PDN_M(nn.Module):
    def __init__(self, last_layer=384):
        super(PDN_S, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=384, kernel_size=4, padding=0, stride=1)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=last_layer, kernel_size=1, padding=0, stride=1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.activation = nn.functional.relu()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        return x