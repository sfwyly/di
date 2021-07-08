# 两种策略联合建模


"""
    残差策略

    每段路直接计算它的残差

    link 部分输入维度：

    Feature:  distance, simple eta, driver id,slice id, link id, link time, link ratio, link current status, link arrival status

    cross 部分

    Feature1 -> Feature2 , cross time

    输入 31 维

    考虑每天天气 和 周几

    输入33 维

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        self.linear1 = nn.Linear(in_features=27, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=512)

        self.linear4 = nn.Linear(in_features=512, out_features=512)

        self.linear5 = nn.Linear(in_features=512, out_features=256)
        self.linear6 = nn.Linear(in_features=256, out_features=128)
        self.linear7 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        # x = F.elu(x, inplace= True)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        # x = F.elu(x, inplace=True)

        return x


class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.linear1 = nn.Linear(in_features=39, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=1024)

        self.linear4 = nn.Linear(in_features=1024, out_features=1024)

        self.linear5 = nn.Linear(in_features=1024, out_features=512)
        self.linear6 = nn.Linear(in_features=512, out_features=256)
        self.linear7 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        # x = F.elu(x, inplace=True)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        # x = F.elu(x, inplace=True)
        return x

def l1_loss(x1, x2):

    return torch.mean(torch.abs(x1 - x2))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):#判断m对象是否属于这些类
        nn.init.xavier_uniform(m.weight.data)
        # nn.init.normal(m.weight.data)
        nn.init.constant(m.bias, 0.1)
