import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class PDN_S(nn.Module):
    def __init__(self, last_layer=384):
        super(PDN_S, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, padding=3, stride=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, padding=3, stride=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(256, last_layer, kernel_size=4, padding=0, stride=1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.activation = F.relu

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
        super(PDN_M, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, padding=3, stride=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, padding=3, stride=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(512, 384, kernel_size=4, padding=0, stride=1)
        self.conv6 = nn.Conv2d(384, last_layer, kernel_size=1, padding=0, stride=1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.avgpool1(x)
        x = self.activation(self.conv2(x))
        x = self.avgpool2(x)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.conv6(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.EncConv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1, stride=2)
        self.EncConv2 = nn.Conv2d(32, 32, kernel_size=4, padding=1, stride=2)
        self.EncConv3 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.EncConv4 = nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.EncConv5 = nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.EncConv6 = nn.Conv2d(64, 64, kernel_size=8, padding=0, stride=1)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.EncConv1(x))
        x = self.activation(self.EncConv2(x))
        x = self.activation(self.EncConv3(x))
        x = self.activation(self.EncConv4(x))
        x = self.activation(self.EncConv5(x))
        x = self.EncConv6(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.DecConv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.DecConv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.DecConv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)

        self.Dropout1 = nn.Dropout(p=0.2)
        self.Dropout2 = nn.Dropout(p=0.2)
        self.Dropout3 = nn.Dropout(p=0.2)
        self.Dropout4 = nn.Dropout(p=0.2)
        self.Dropout5 = nn.Dropout(p=0.2)
        self.Dropout6 = nn.Dropout(p=0.2)

        self.activation = F.relu

    def forward(self, x):
        x = F.interpolate(x, size=3, mode='bilinear')
        x = self.activation(self.DecConv1(x))
        x = self.Dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.activation(self.DecConv2(x))
        x = self.Dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = self.activation(self.DecConv3(x))
        x = self.Dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = self.activation(self.DecConv4(x))
        x = self.Dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = self.activation(self.DecConv5(x))
        x = self.Dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = self.activation(self.DecConv6(x))
        x = self.Dropout6(x)
        x = F.interpolate(x, size=64, mode='bilinear')
        x = self.activation(self.DecConv7(x))
        x = self.DecConv8(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()
    
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PDN_M()
    model = model.to(device)
    summary(model, (3, 256, 256))
