import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import MyDataset
from fedavg import FedAvgTrainer
from init_args import add_args
from ResNet_model import BuildUnet

if __name__ == "__main__":
    args = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = BuildUnet()

    img1, label1 = [], []
    f = open("datatxt/img1.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.transpose(img_item, (2, 0, 1))
        img1.append(img_item)

    f = open("datatxt/lab1.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.reshape(img_item, (1, img_item.shape[0], img_item.shape[1]))
        label1.append(img_item)

    img2, label2 = [], []
    f = open("datatxt/img2.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.transpose(img_item, (2, 0, 1))
        img2.append(img_item)

    f = open("datatxt/lab2.txt", 'r')
    for item in f.readlines():
        img_item = np.array(np.array(np.load(item.rstrip()), dtype='float32'))
        img_item = np.reshape(img_item, (1, img_item.shape[0], img_item.shape[1]))
        label2.append(img_item)

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

    train_dataset = MyDataset(img1, label1, augmentation=True)
    dataloader1 = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = MyDataset(test_img, test_img, augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    train_dataset = MyDataset(img2, label2, augmentation=True)
    dataloader2 = DataLoader(train_dataset, batch_size=4, shuffle=True)
    global_dataset = [dataloader1, dataloader2]
    test_data = [test_loader, test_loader]
    trainer = FedAvgTrainer(global_dataset, test_data, model, device, args)
    trainer.train()
