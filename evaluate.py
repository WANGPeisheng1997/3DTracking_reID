import torch
import numpy as np


def evaluate(query_feature, query_label, query_camera, gallery_feature, gallery_label, gallery_camera):
    # 计算query的feature与gallery中每一张图片feature的点积，只需将galley feature乘以query feature的转置
    query_feature_transpose = query_feature.view(-1, 1)
    similarity = torch.matmul(gallery_feature, query_feature_transpose)
    similarity = similarity.squeeze(1).cpu()
    similarity = similarity.numpy()
    # similarity（即cosine distance）从大到小排序后对应原gallery中编号
    index = np.argsort(similarity)
    index = index[::-1]
    print(index)
    # 在gallery中找到与query的id和camera相同的图片
    query_index = np.argwhere(gallery_label==query_label)
    camera_index = np.argwhere(gallery_camera==query_camera)
    print(query_index)
    print(camera_index)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gallery_label==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def test_market(query_feature, query_label, query_camera, gallery_feature, gallery_label, gallery_camera):
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_camera[i], gallery_feature, gallery_label,
                                   gallery_camera)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))