import numpy as np
from torch.utils.data import DataLoader

from LossFunction import dice_function
from datasets import MyDataset


def iou(true_mask, predicted_mask):
    intersection = np.sum(true_mask * predicted_mask)
    union = np.sum(true_mask) + np.sum(predicted_mask) - intersection
    iou = intersection / (union + 1e-8)
    return iou


def precision(true_mask, predicted_mask):
    true_positive = np.sum(true_mask * predicted_mask)
    false_positive = np.sum(predicted_mask) - true_positive
    precision = true_positive / (true_positive + false_positive + 1e-8)
    return precision


def recall(true_mask, predicted_mask):
    true_positive = np.sum(true_mask * predicted_mask)
    false_negative = np.sum(true_mask) - true_positive
    recall = true_positive / (true_positive + false_negative + 1e-8)
    return recall


def test_net(net, device):
    test_img, test_lab = [], []
    f = open("datatxt/val_img.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.transpose(img_item, (2, 0, 1))
        test_img.append(img_item)

    f = open("datatxt/val_lab.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.reshape(img_item, (1, img_item.shape[0], img_item.shape[1]))
        test_lab.append(img_item)

    test_dataset = MyDataset(test_img, test_lab)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4)
    re_list, dice_list, precision_list, iou_list = [], [], [], []
    net = net.to(device)
    for img, lab in test_dataloader:
        img, lab = img.to(device), lab.to(device)
        pred = net(lab)
        re_list.append(recall(lab, pred))
        dice_list.append(dice_function(lab, pred))
        precision_list.append(precision(lab, pred))
        iou_list.append(iou(lab, pred))
    re = np.mean(re_list)
    dice = np.mean(dice_list)
    prec = np.mean(precision_list)
    io = np.mean(iou_list)
    print("recall:{0:.4f},dice:{1:.4f},precision:{2:.4f},iou:{3:.4f}".format(re, dice, prec, io))
