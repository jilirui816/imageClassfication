"""
CNN案例
卷积层：
提取图像局部特征->特征图（Feature Map）,计算方式：N=(W-F+2P)/S+1
每个卷积核为一个神经元


池化层：
降维，有最大池化和平均池化

池化尽在HW上做调整，通道不改变
"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader

# 每批样本数
BATCH_SIZE = 8
#准备数据
def create_dataset():
#     获取数据集 参1 数据集路径 参2 是否是训练集 参2 数据预处理->张量数据 参4：是否互联网下载
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=False)
    # 获取测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(),download=False)

#     返回数据集
    return train_dataset, test_dataset



# 搭建神经网络

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层（加入BN，增加卷积核数量，调整padding）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层（加入Dropout防止过拟合）
        self.linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=512)
        self.drop1 = nn.Dropout(p=0.5)  # 50%概率丢弃神经元
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.drop2 = nn.Dropout(p=0.5)
        self.output = nn.Linear(in_features=256, out_features=10)

        # 激活函数改为LeakyReLU
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # 卷积层：Conv → BN → 激活 → 池化
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 展平（适配新的卷积层输出维度）
        x = x.reshape(x.shape[0], -1)

        # 全连接层：Linear → 激活 → Dropout
        x = self.act(self.linear1(x))
        x = self.drop1(x)

        x = self.act(self.linear2(x))
        x = self.drop2(x)

        # 输出层（CrossEntropyLoss包含Softmax，无需手动加）
        return self.output(x)








# 模型训练

def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    model.train()
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        #定义变量记录 总损失，总样本数据量，预测正确样本个数，训练时间
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for x,y in dataloader:

            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()

            #统计预测正确的样本个数

            total_correct+=torch.argmax(y_pred,dim=-1).eq(y).sum()
            total_loss+=loss.item()*len(y)
            total_samples+=len(y)

#         保存模型
    torch.save(model.state_dict(), "./model/image_model.pth")


#模型测试
def test(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
#     加载模型参数
    model.load_state_dict(torch.load("./model/image_model.pth"))
    total_correct = 0
    total_samples = 0

    for x,y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        total_correct+=y_pred.eq(y).sum().item()
        total_samples+=len(y)
    print(f'正确率：{total_correct/total_samples:.2f}')



# 册数
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    model = ImageModel()
    # 训练
    train(train_dataset)
    test(test_dataset)
