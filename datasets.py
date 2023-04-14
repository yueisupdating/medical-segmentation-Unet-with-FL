import random
import cv2
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, imgs_list, label_list, augmentation=False):
        super().__init__()
        self.label_list = label_list
        self.imgs_list = imgs_list
        self.augmentation = augmentation

    def augment(self, item, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        if self.augmentation:
            flip = cv2.flip(item, flipCode)
            return flip
        return item

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):  # index为索引
        img = self.imgs_list[index]
        label = self.label_list[index]
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            img = self.augment(img, flipCode)
        label = self.augment(label, flipCode)
        return img, label
