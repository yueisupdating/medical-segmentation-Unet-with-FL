import torch.nn as nn
import torch
from torchsummary import summary


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.c = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, inputs):
        x = self.c(inputs)
        s = self.bottleneck(inputs)
        return x + s


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.l5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.l1(inputs)
        x2 = self.l2(inputs)
        x3 = self.l3(inputs)
        x4 = self.l4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.l5(x)
        return y


class AttentionLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[0], in_c[1], kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[1], in_c[1], kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[1], in_c[1], kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y


class DecoderLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a = AttentionLayer(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = ConvLayer(in_c[0] + in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a(g, x)
        d = self.up(d)
        d = torch.cat([d, g], dim=1)
        d = self.c1(d)
        return d


class BuildUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvLayer(3, 16, stride=1)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = ConvLayer(16, 32, stride=1)
        self.m2 = nn.MaxPool2d(2)
        self.c3 = ConvLayer(32, 64, stride=1)
        self.m3 = nn.MaxPool2d(2)
        self.c4 = ConvLayer(64, 128, stride=1)
        self.b1 = ASPP(128, 256)
        self.d1 = DecoderLayer([64, 256], 128)
        self.d2 = DecoderLayer([32, 128], 64)
        self.d3 = DecoderLayer([16, 64], 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        m1 = self.m1(c1)
        c2 = self.c2(m1)
        m2 = self.m2(c2)
        c3 = self.c3(m2)
        m3 = self.m3(c3)
        c4 = self.c4(m3)
        b1 = self.b1(c4)
        u1 = self.d1(c3, b1)
        u2 = self.d2(c2, u1)
        u3 = self.d3(c1, u2)
        out = self.output(u3)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BuildUnet()
    model.to(device=device)
    summary(model, input_size=(3, 240, 240))
