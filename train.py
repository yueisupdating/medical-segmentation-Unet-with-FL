import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from medpy.metric.binary import dc, precision

from LossFunction import loss_function
from datasets import MyDataset
from transformerUnet import TransUNet

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


def train_net(net, device, epochs=50, batch_size=4, lr=1e-4):
    best_loss = -0x3f3f3f3f
    plt_loss = []
    plt_trainloss = []
    # 加载训练集
    train_dataset = MyDataset(img1, label1, augmentation=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4)
    test_dataset = MyDataset(test_img, test_lab, augmentation=False)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=4)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)  # 优化器

    # 训练模型
    for epoch in range(epochs):
        print('第%d轮' % epoch)
        train_accuracy = 0
        train_dice = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            pred = net(images)
            loss = loss_function(pred, labels)
            loss.backward()
            train_dice += loss
            train_accuracy += precision(pred.detach().numpy(), labels.detach().numpy())
            optimizer.step()
        train_accuracy /= len(train_loader)
        train_dice /= len(train_loader)
        print('train_dice:{:.4f},train_accuracy: {:.4f}'.format(train_dice, train_accuracy))
        plt_trainloss.append(train_accuracy)

        # 评估模型
        total = 0
        test_dice = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                pred = model(images)
                test_dice += dc(pred.detach().numpy(), labels.detach().numpy())
                total += precision(pred.detach().numpy(), labels.detach().numpy())
        total_loss = total / len(test_loader)
        test_dice /= len(test_loader)
        print('test_dice:{:4.f},test_accuracy: {:.4f}'.format(test_dice, total_loss))
        plt_loss.append(total_loss)
        if total_loss > best_loss:
            torch.save(net.state_dict(), 'model.pth')
    plt.plot(list(range(1, len(plt_loss) + 1)), plt_loss, list(range(1, len(plt_trainloss) + 1)), plt_trainloss)
    plt.show()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1。
    model = TransUNet(img_dim=240, in_channels=3, out_channels=128, head_num=8, mlp_dim=512, block_num=8, patch_dim=16,
                      class_num=1)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 指定训练集地址，开始训练
    train_net(model, device)
