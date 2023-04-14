import torch
import torch.nn as nn
from einops import rearrange
from torchsummary import summary

from vit import ViT


class InputBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.input_channel, self.out_channel = input_channel, out_channel
        self.c1 = nn.Conv2d(in_channels=input_channel, out_channels=out_channel, kernel_size=3)
        self.bn = nn.BatchNorm2d(self.input_channel)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.bn(out)
        out = self.rl(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, input_channel, out_channel, stride=1, padding=0):
        super().__init__()
        self.input, self.out = input_channel, out_channel
        self.c = nn.Conv2d(self.input, self.out, kernel_size=1, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(self.out)

    def forward(self, x):
        out = self.c(x)
        out = self.bn1(out)
        return out


class Conv(nn.Module):
    def __init__(self, input_channel, out_channel, stride=1):
        super().__init__()
        self.input_channel, self.out_channel = input_channel, out_channel
        self.c1 = nn.Conv2d(self.input_channel, self.out_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.c2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.c3 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.bottle = Bottleneck(self.input_channel, self.out_channel, stride)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.bn1(out)
        out = self.c2(out)
        out = self.bn2(out)
        out = self.c3(out)
        out = self.bn3(out)
        out += self.bottle(x)
        out = self.rl(out)
        return out


class encoder(nn.Module):
    def __init__(self, inputs, out, img_dim, patch_dim, head_num, mlp_dim, block_num):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, out, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = Conv(out, out * 2, stride=2)
        self.encoder2 = Conv(out * 2, out * 4, stride=2)
        self.encoder3 = Conv(out * 4, out * 8, stride=2)
        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(img_dim=self.vit_img_dim, in_channels=out * 8, embedding_dim=out * 8, head_num=head_num,
                       mlp_dim=mlp_dim,
                       block_num=block_num, patch_dim=1, classification=False)

        # mlp_dim:隐藏层中神经元的数量,block_num:展成一维向量后图像尺寸
        self.conv2 = nn.Conv2d(out * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x, x1, x2, x3


class DoubleConv(nn.Module):
    def __init__(self, inputs, out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()
        self.decoder1 = UpSample(out_channels * 8, out_channels * 2)
        self.decoder2 = UpSample(out_channels * 4, out_channels)
        self.decoder3 = UpSample(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = UpSample(int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        return x


class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, img_dim, patch_dim, head_num, mlp_dim, block_num, class_num=1):
        super().__init__()
        self.encoder = encoder(img_dim=img_dim, inputs=in_channels, out=out_channels, head_num=head_num,
                               mlp_dim=mlp_dim, block_num=block_num, patch_dim=patch_dim)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        return x


if __name__ == '__main__':
    model = TransUNet(img_dim=240, in_channels=3, out_channels=128, head_num=8, mlp_dim=512, block_num=8,
                      patch_dim=16, class_num=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    summary(model, input_size=(3, 240, 240))
