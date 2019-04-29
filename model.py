import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, class_num, drop_rate=0.5):
        super(Model, self).__init__()
        resnet_model = models.resnet50(pretrained=True)
        self.model = resnet_model
        self.dense = torch.nn.Sequential()
        self.dense.add_module("fc1", nn.Linear(2048, 512))
        self.dense.add_module("bn", nn.BatchNorm1d(512))
        if drop_rate > 0:
            self.dense.add_module("drop", nn.Dropout(p=drop_rate))
        self.dense.add_module("fc2", nn.Linear(512, class_num))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # 去除最后一层fc，因此x的大小是[n,2048,1,1]
        x = x.view(x.size(0), x.size(1))
        # 只保留前两个维度[n,2048]
        x = self.dense(x)
        return x


if __name__ == '__main__':
    net = Model(751)
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
