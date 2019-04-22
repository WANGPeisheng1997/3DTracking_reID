from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import time
import os
from model import Model

# 设定参数
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir', default='/data_set', type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

opt = parser.parse_args()

# 设定gpu
use_gpu = torch.cuda.is_available()
gpu_ids_str = opt.gpu_ids.split(',')
gpu_ids = []

for str_id in gpu_ids_str:
    gpu_id = int(str_id)
    if gpu_id >= 0:
        gpu_ids.append(gpu_id)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# 图像预处理（随机填充+裁剪+水平翻转）

transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_test_list = [
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_dir = opt.data_dir
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose(transform_train_list))

# 数据读取

data_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=True, num_workers=8, pin_memory=True)
dataset_sizes = len(image_datasets[x])
classes_num = len(image_datasets.classes)

# 训练模型

y_loss = []
y_err = []


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    train_start_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        scheduler.step()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0.0

        for data in data_loaders:
            inputs, labels = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < opt.batchsize:  # 剩余大小不足一个batch时跳过
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                labels = Variable(labels.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # 梯度初始化
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算loss
            _, predictions = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # 训练时反向传播
            loss.backward()
            optimizer.step()

            # 统计loss和acc
            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(predictions == labels.data))

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects / dataset_sizes

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            y_loss.append(epoch_loss)
            y_err.append(1.0 - epoch_acc)

            # 保存模型
            last_model_wts = model.state_dict()
            if epoch % 10 == 9:
                save_network(model, epoch)

        time_elapsed = time.time() - train_start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - train_start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # 保存并返回最后一次模型
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


# 保存模型
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


# 加载模型
model = Model(classes_num, opt.droprate, opt.stride)
print(model)

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# 设置学习率递减
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

model = model.cuda()
criterion = nn.CrossEntropyLoss()
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=60)