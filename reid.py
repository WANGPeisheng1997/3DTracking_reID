import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from PIL import Image

import os
from model import Model

# 设定参数
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--data_dir', default='gallery', type=str, help='the directory of the data set')
parser.add_argument('--use_final_feature', action='store_true', help='use the feature after fc2')

opt = parser.parse_args()

classes_num = 751
if opt.use_final_feature:
    feature_dimension = 751
else:
    feature_dimension = 512

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


def load_network(network):
    save_path = os.path.join('model', 'epoch_%s.model' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def extract_feature(model, image):
    input_img = Variable(image.cuda())
    input_img = input_img.view(1, 3, 256, 128)
    outputs = model(input_img)
    feature = outputs.data.cpu().float()
    return feature


def rank(query, gallery):
    # query and gallery: {id:feature, ...}
    for id in query:
        print(id)
        query_feature = query[id][0]
        similarity = {}
        for g_id in gallery:
            gallery_feature = gallery[g_id][0]
            sim = query_feature.dot(gallery_feature)
            similarity[g_id] = sim
        print(similarity)


# 加载模型并提取特征

model_structure = Model(classes_num)
model = load_network(model_structure)
if not opt.use_final_feature:
    model.dense.fc2 = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

# 提取所有图像特征

total_frame = 29
total_id = 16
each_frame_features = []
for frame in range(total_frame):
    print("frame:%d" % frame)
    gallery_features = {}
    for view in ["l", "m", "r"]:
        image_directory = os.path.join(opt.data_dir, view, str(frame))
        features = {}
        for id in range(total_id):
            image_path = os.path.join(image_directory, str(id) + ".bmp")
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = data_transforms(image)
                with torch.no_grad():
                    feature = extract_feature(model, image)
                features[id] = feature
        gallery_features[view] = features
    each_frame_features.append(gallery_features)
    rank(gallery_features["m"], gallery_features["l"])


# print(each_frame_features)
