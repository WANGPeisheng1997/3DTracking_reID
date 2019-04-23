import os
from shutil import copyfile

# market-1501 数据集路径
market_path = '../Market'

# 整理后的数据集路径
save_path = 'data_set'
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# 按照id对原数据集中的数据进行分类整理，便于使用ImageFolder
def classify_from_source_to_destination(src_path, dst_path):
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    for root, dirs, files in os.walk(src_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            id = name.split('_')
            src_dir = src_path + '/' + name
            dst_dir = dst_path + '/' + id[0]
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            copyfile(src_dir, dst_dir + '/' + name)


# 训练集 751个ID
train_path = market_path + '/bounding_box_train'
train_save_path = save_path + '/train'
classify_from_source_to_destination(train_path, train_save_path)

# 测试集 750个ID
test_path = market_path + '/bounding_box_test'
test_save_path = save_path + '/test'
classify_from_source_to_destination(test_path, test_save_path)

# query 3368张
query_path = market_path + '/query'
query_save_path = save_path + '/query'
classify_from_source_to_destination(query_path, query_save_path)