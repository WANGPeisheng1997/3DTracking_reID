import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from PIL import Image
from read_3d import get_nearby_info

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
    feature = outputs.data.cpu().float()[0]
    # L2 norm 82.4%->84.7%
    norm2 = feature.norm()
    feature = feature.div(norm2)
    return feature


def max_sim_id(dict):
    max_sim_dict = {}
    for id in dict:
        max_sim_dict[id] = dict[id][0][1]
    max_sim_dict = sorted(max_sim_dict.items(), key=lambda item: item[1], reverse=True)
    return max_sim_dict[0][0]


def rank(query, gallery):
    # query and gallery: {id:feature, ...}
    total = 0
    rank1_correct = 0
    rank_result_dict = {}
    match_correct = 0
    for q_id in query:
        print(q_id)
        query_feature = query[q_id]
        similarity = {}
        for g_id in gallery:
            gallery_feature = gallery[g_id]
            sim = query_feature.dot(gallery_feature)
            similarity[g_id] = sim
        # print(similarity)
        rank_result = sorted(similarity.items(), key=lambda item: item[1], reverse=True)
        rank_result_dict[q_id] = rank_result
        print(rank_result)
        total += 1
        if rank_result[0][0] == q_id:
            rank1_correct += 1

    match_dict = {}
    for i in range(len(query)):
        max_id = max_sim_id(rank_result_dict)
        # 第max_id个query匹配了第rank_result_dict[max_id][0][0]个gallery
        match_id = rank_result_dict[max_id][0][0]
        match_dict[max_id] = match_id
        # 删除相似矩阵的max_id行match_id列
        del(rank_result_dict[max_id])
        for id in rank_result_dict:
            similarity = rank_result_dict[id]
            for j in range(len(similarity)):
                if similarity[j][0] == match_id:
                    del(similarity[j])
                    break
            rank_result_dict[id] = similarity

    print(match_dict)
    for id in match_dict:
        if id == match_dict[id]:
            match_correct += 1

    return torch.Tensor([total, rank1_correct, match_correct])


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
result = torch.FloatTensor([0, 0, 0])
# for frame in range(total_frame):
#     print("frame:%d" % frame)
#     gallery_features = {}
#     for view in ["l", "m", "r"]:
#         image_directory = os.path.join(opt.data_dir, view, str(frame))
#         features = {}
#         for id in range(total_id):
#             image_path = os.path.join(image_directory, str(id) + ".bmp")
#             if os.path.exists(image_path):
#                 image = Image.open(image_path).convert('RGB')
#                 image = data_transforms(image)
#                 with torch.no_grad():
#                     feature = extract_feature(model, image)
#                 features[id] = feature
#         gallery_features[view] = features
#     each_frame_features.append(gallery_features)
#     result += rank(gallery_features["m"], gallery_features["l"])
#     result += rank(gallery_features["m"], gallery_features["r"])
#     total, rank1_correct, match_correct = result[0], result[1], result[2]
#
#
# print("cross-views re-id")
# print("rank1:%.3f, %d/%d" % (rank1_correct/total, rank1_correct, total))
# print("match accuracy:%.3f, %d/%d" % (match_correct/total, match_correct, total))

range_person_array, correct_answer_array = get_nearby_info()


for yuzhi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]:

    correct = 0
    total = 0

    for frame in range(total_frame - 1):
        print("frame:%d and %d" % (frame, frame + 1))

        for view in ["l", "m", "r"]:

            current_image_directory = os.path.join(opt.data_dir, view, str(frame))
            next_image_directory = os.path.join(opt.data_dir, view, str(frame + 1))

            current_features = {}
            for id in range(total_id):
                image_path = os.path.join(current_image_directory, str(id) + ".bmp")
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    image = data_transforms(image)
                    with torch.no_grad():
                        feature = extract_feature(model, image)
                    current_features[id] = feature

            next_features = {}
            for id in range(total_id):
                image_path = os.path.join(next_image_directory, str(id) + ".bmp")
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    image = data_transforms(image)
                    with torch.no_grad():
                        feature = extract_feature(model, image)
                    next_features[id] = feature


            for id in current_features:
                nearby_person_ids = range_person_array[frame][id]
                if nearby_person_ids == []:
                    match_id = -1
                else:
                    similarity = {}
                    for n_id in nearby_person_ids:
                        sim = current_features[id].dot(next_features[n_id])
                        similarity[n_id] = sim

                    rank_result = sorted(similarity.items(), key=lambda item: item[1], reverse=True)
                    # print(rank_result)
                    if rank_result[0][1] >= yuzhi:
                        match_id = rank_result[0][0]
                    else:
                        match_id = -1

                total += 1
                correct_answer = id if correct_answer_array[frame][id] else -1
                if match_id == correct_answer:
                    correct += 1
                else:
                    print("match_id:" + str(match_id) + " correct_id:", correct_answer)



    print("cross-frames re-id")
    print("match accuracy:%.3f, %d/%d" % (correct/total, correct, total))

    # print(each_frame_features)
