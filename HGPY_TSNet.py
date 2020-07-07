import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pandas as pd
import seaborn as sns
import numpy as np
import cv2 as cv


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv1_2 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv1_0 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv2_1 = nn.Conv2d(24, 64, 5, padding=2)
        self.conv2_2 = nn.Conv2d(24, 64, 5, padding=2)
        self.conv2_0 = nn.Conv2d(24, 64, 5, padding=2)
        self.conv3_1 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv3_0 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv4_1 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv4_2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv4_0 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv5_1 = nn.Conv2d(96, 64, 3, padding=1)
        self.conv5_2 = nn.Conv2d(96, 64, 3, padding=1)
        self.conv5_0 = nn.Conv2d(96, 64, 3, padding=1)
        self.fc1_1 = nn.Linear(64 * 8 * 8, 128)
        self.fc1_2 = nn.Linear(64 * 8 * 8, 128)
        self.fc1_0 = nn.Linear(64 * 8 * 8, 128)
        self.fc2_1 = nn.Linear(256, 512)
        self.fc2_0 = nn.Linear(256, 512)
        self.fc3_1 = nn.Linear(512, 512)
        self.fc3_0 = nn.Linear(512, 512)
        self.fc4_1 = nn.Linear(512, 2)
        self.fc4_0 = nn.Linear(512, 2)
        self.fc5 = nn.Linear(4, 2)

    def subNet_0(self, x):
        in_size = x.size(0)
        # [in,3,64,64]
        out = F.leaky_relu(self.conv1_0(x))
        out = F.max_pool2d(out, 2, 2)
        # [in,24,32,32]
        out = F.leaky_relu(self.conv2_0(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,16,16]
        out = F.leaky_relu(self.conv3_0(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv4_0(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv5_0(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,8,8]
        out = out.view(in_size, -1)
        # [in,64*8*8]
        out = F.leaky_relu(self.fc1_0(out))
        # [in,128]
        return out

    def subNet_1(self, x):
        in_size = x.size(0)
        # [in,3,64,64]
        out = F.leaky_relu(self.conv1_1(x))
        out = F.max_pool2d(out, 2, 2)
        # [in,24,32,32]
        out = F.leaky_relu(self.conv2_1(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,16,16]
        out = F.leaky_relu(self.conv3_1(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv4_1(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv5_1(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,8,8]
        out = out.view(in_size, -1)
        # [in,64*8*8]
        out = F.leaky_relu(self.fc1_1(out))
        # [in,128]
        return out

    def subNet_2(self, x):
        in_size = x.size(0)
        # [in,3,64,64]
        out = F.leaky_relu(self.conv1_2(x))
        out = F.max_pool2d(out, 2, 2)
        # [in,24,32,32]
        out = F.leaky_relu(self.conv2_2(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,16,16]
        out = F.leaky_relu(self.conv3_2(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv4_2(out))
        # [in,96,16,16]
        out = F.leaky_relu(self.conv5_2(out))
        out = F.max_pool2d(out, 2, 2)
        # [in,64,8,8]
        out = out.view(in_size, -1)
        # [in,64*8*8]
        out = F.leaky_relu(self.fc1_2(out))
        # [in,128]
        return out

    def forward(self, x, y):
        out_01 = self.subNet_0(x)
        out_1 = self.subNet_1(x)
        out_02 = self.subNet_0(y)
        out_2 = self.subNet_2(y)
        out_l = F.leaky_relu(self.fc4_0(
            F.leaky_relu(self.fc3_0(
                F.leaky_relu(self.fc2_0(torch.cat((out_01, out_02), dim=1)))
            ))
        ))
        out_r = F.leaky_relu(self.fc4_1(
            F.leaky_relu(self.fc3_1(
                F.leaky_relu(self.fc2_1(torch.cat((out_1, out_2), dim=1)))
            ))
        ))
        out_top = F.leaky_relu(self.fc5(torch.cat((out_l, out_r), dim=1)))

        return out_01, out_02, out_1, out_2, out_l, out_r, out_top


def contrastive_loss(out_1, out_2, labels):
    D = F.pairwise_distance(out_1, out_2, p=2)
    Q = 50
    loss = (2 / Q) * labels * (D ** 2) + (1 - labels) * 2 * Q * torch.exp((-2.77 / Q) * D)
    return loss


def train_sol(out_01, out_02, out_1, out_2, out_l, out_r, out_top, BATCH_SIZE, positive):
    if positive == 1:
        labels = torch.ones([BATCH_SIZE], dtype=torch.long)
    else:
        labels = torch.zeros([BATCH_SIZE], dtype=torch.long)
    loss_cel_1 = nn.CrossEntropyLoss()
    loss_cel_2 = nn.CrossEntropyLoss()
    loss_cel_3 = nn.CrossEntropyLoss()
    loss_con_l = contrastive_loss(out_01, out_02, labels)
    loss_con_r = contrastive_loss(out_1, out_2, labels)
    loss_en_l = loss_cel_1(out_l, labels)
    loss_en_r = loss_cel_2(out_r, labels)
    loss_en_m = loss_cel_3(out_top, labels)
    loss = loss_en_l + loss_en_m + loss_en_r + 0.01 * loss_con_l + 0.01 * loss_con_r
    return torch.mean(loss)


def train():
    total_images = 70000
    train_data_co = np.zeros((total_images, 3, 64, 64), dtype='float32')
    train_data_ir = np.zeros((total_images, 3, 64, 64), dtype='float32')
    for i in range(total_images):
        img_path_co = "nirscene1/{:0>6d}_rgb.png".format(i)
        train_data_co[i, :, :, :] = np.transpose(cv.imread(img_path_co), (2, 0, 1))
        img_path_ir = "nirscene1/{:0>6d}_nir.png".format(i)
        train_data_ir[i, :, :, :] = np.transpose(cv.imread(img_path_ir), (2, 0, 1))
    for i in range(len(train_data_co)):
        image = train_data_co[i, :, :, :]
        result = np.zeros_like(image)
        cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        train_data_co[i, :, :, :] = result
        image = train_data_ir[i, :, :, :]
        result = np.zeros_like(image)
        cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        train_data_ir[i, :, :, :] = result
    print("数据读取完成")

    model.train()
    train_data_co = torch.from_numpy(train_data_co).to(DEVICE)
    train_data_ir = torch.from_numpy(train_data_ir).to(DEVICE)

    for i in range(total_images // BATCH_SIZE - 1):
        rand_num = np.zeros(BATCH_SIZE)
        for j in range(BATCH_SIZE):
            rand_1 = i * BATCH_SIZE
            while i * BATCH_SIZE <= rand_1 < (i + 1) * BATCH_SIZE:
                rand_1 = random.randint(0, total_images - 1)
            rand_num[j] = rand_1

        optimizer.zero_grad()
        out_01, out_02, out_1, out_2, out_l, out_r, out_top = model(
            train_data_co[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :],
            train_data_ir[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
        )
        loss_p = train_sol(out_01, out_02, out_1, out_2, out_l, out_r, out_top, BATCH_SIZE, 1)
        loss_p.backward()
        optimizer.step()

        optimizer.zero_grad()
        out_01, out_02, out_1, out_2, out_l, out_r, out_top = model(
            train_data_co[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :],
            train_data_ir[rand_num, :, :, :]
        )
        loss_n = train_sol(out_01, out_02, out_1, out_2, out_l, out_r, out_top, BATCH_SIZE, 0)
        loss_n.backward()
        optimizer.step()

        print("Train Epoch:{} [{}/{} ({:.2f}%)]\tLoss_p:{:.6f}\tLoss_n:{:.6f}".format(
            epoch, i * BATCH_SIZE, total_images, 100 * i * BATCH_SIZE / total_images, loss_p.item(), loss_n.item()
        ))


def test():
    total_images = 4396
    train_data_co = np.zeros((total_images, 3, 64, 64), dtype='float32')
    train_data_ir = np.zeros((total_images, 3, 64, 64), dtype='float32')
    for i in range(total_images):
        img_path_co = "nirscene1/{:0>6d}_rgb.png".format(i + 70000)
        train_data_co[i, :, :, :] = np.transpose(cv.imread(img_path_co), (2, 0, 1))
        img_path_ir = "nirscene1/{:0>6d}_nir.png".format(i + 70000)
        train_data_ir[i, :, :, :] = np.transpose(cv.imread(img_path_ir), (2, 0, 1))
    for i in range(len(train_data_co)):
        image = train_data_co[i, :, :, :]
        result = np.zeros_like(image)
        cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        train_data_co[i, :, :, :] = result
        image = train_data_ir[i, :, :, :]
        result = np.zeros_like(image)
        cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        train_data_ir[i, :, :, :] = result

    model.eval()
    train_data_co = torch.from_numpy(train_data_co).to(DEVICE)
    train_data_ir = torch.from_numpy(train_data_ir).to(DEVICE)

    correct_p = 0
    correct_n = 0
    total = 0
    with torch.no_grad():
        for i in range(total_images - 1):
            out_01, out_02, out_1, out_2, out_l, out_r, out_top = model(
                train_data_co[np.newaxis, i, :, :, :],
                train_data_ir[np.newaxis, i, :, :, :]
            )
            correct_p += sum(out_top[:, 0] < out_top[:, 1]).item()

            rand_1 = i
            while rand_1 == i:
                rand_1 = random.randint(0, total_images - 1)

            out_01, out_02, out_1, out_2, out_l, out_r, out_top = model(
                train_data_co[np.newaxis, i, :, :, :],
                train_data_ir[np.newaxis, rand_1, :, :, :]
            )
            correct_n += sum(out_top[:, 0] > out_top[:, 1]).item()

            total += len(out_top) * 2

    print("\nTest set: Average loss:({}+{})/{}={:.4f}\n".format(
        correct_p, correct_n, total, (correct_p + correct_n) / total
    ))
    return (correct_p + correct_n) / total


if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCHS = 150
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load('./checkpoint/state_co_{}.pth'.format(150), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state'])
    print("epoch={},acc={}".format(checkpoint['epoch'], checkpoint['acc']))

    '''for epoch in range(1, EPOCHS + 1):
        print('===> Training models...')
        train()
        print('===> Testing models...')
        acc = test()
        print('===> Saving models...')
        state = {
            'state': model.state_dict(),
            'epoch': epoch,  # 将epoch一并保存
            'acc': acc
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/state_co_{}.pth'.format(epoch))'''
