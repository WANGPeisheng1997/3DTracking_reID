from PIL import Image, ImageDraw
from scipy.io import loadmat
import os


def euclidean_distance(pos1, pos2):
    return pow((pow((pos1[0] - pos2[0]), 2) + pow((pos1[1] - pos2[1]), 2) + pow((pos1[2] - pos2[2]), 2)), 0.5)


def get_nearby_info():
    m = loadmat("AX3D.mat")
    pos_dense_array = []
    for frame in range(0, 29):
        pos_array = m["Ax_3d"][frame]
        pos_dict = {}
        for id in range(0, 16):
            pos_3d = pos_array[id]
            if pos_3d.shape == (3, 1):
                pos_dict[id] = pos_3d
        pos_dense_array.append(pos_dict)

    # print(pos_dense_array)

    # max distance: 209.77
    range_person_array = []
    for frame in range(0, 28):
        current_frame_dict = pos_dense_array[frame]
        next_frame_dict = pos_dense_array[frame+1]
        range_person_dict = {}
        for i in current_frame_dict:
            range_person = []
            for j in next_frame_dict:
                distance = euclidean_distance(current_frame_dict[i], next_frame_dict[j])
                if distance < 250:
                    range_person.append(j)
            range_person_dict[i] = range_person
        range_person_array.append(range_person_dict)

    correct_answer_array = []
    for frame in range(0, 28):
        current_frame_dict = pos_dense_array[frame]
        answer_dict = {}
        for id in current_frame_dict:
            if(m["Ax_3d"][frame + 1][id].shape==(3,1)):
                answer_dict[id] = True
            else:
                answer_dict[id] = False
        correct_answer_array.append(answer_dict)

    #range_person_array[frame][id]表明在第frame+1帧中有哪些id位于frame帧id附近
    # print(range_person_array)
    #correct_answer_array[frame][id]表明在第frame+1帧中是否存在id
    # print(correct_answer_array)

    return range_person_array, correct_answer_array