from torch import nn
import torch


def dice_function(y_pred, y_true, smooth=1e-2):
    intersection = torch.sum(y_pred * y_true, dim=[1, 2, 3])
    union = torch.sum(y_pred, dim=[1, 2, 3]) + torch.sum(y_true, dim=[1, 2, 3])
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - torch.mean(dice)


def loss_function(y_pred, y_true):
    bce = nn.CrossEntropyLoss()
    return 0.5 * dice_function(y_pred, y_true) + 0.5 * bce(y_pred, y_true)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(4, 1, 240, 240)
    y = torch.rand(4, 1, 240, 240)
    print(loss_function(x, y))
