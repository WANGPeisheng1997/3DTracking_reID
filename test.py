import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

import os
from model import Model
from evaluate import test_market

# 设定参数
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--data_dir', default='data_set', type=str, help='the directory of the data set')
parser.add_argument('--batchsize', default=256, type=int, help='batch size')

opt = parser.parse_args()

classes_num = 751

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

# 图像预处理
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据读取
image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data_dir, x), data_transforms) for x in ['test', 'query']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['test', 'query']}
class_names = image_datasets['query'].classes


def load_network(network):
    save_path = os.path.join('model', 'epoch_%s.model' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def fliplr(img): # 左右翻转
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloader):
    dataset_size = len(dataloader.dataset)
    features = torch.FloatTensor()
    extracted_count = 0
    for data in dataloader:
        img, label = data
        n, c, h, w = img.size()
        extracted_count += n
        print("Progress: %d/%d" % (extracted_count, dataset_size))
        ff = torch.FloatTensor(n, classes_num).zero_()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu().float()
            ff = ff + f

            # L2norm
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


# 获取gallery和query各图像信息

gallery_path = image_datasets['test'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

# 加载模型并提取特征

model_structure = Model(classes_num)
model = load_network(model_structure)
# model.classifier.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

with torch.no_grad():
    print("Extracting features from gallery!")
    gallery_feature = extract_feature(model, data_loaders['test'])
    print("Extracting features from query!")
    query_feature = extract_feature(model, data_loaders['query'])
    print("Extracting complete!")

test_market(query_feature, query_label, query_cam, gallery_feature, gallery_label, gallery_cam)
