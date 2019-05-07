from PIL import Image, ImageDraw
from scipy.io import loadmat
import os

m = loadmat("MA_n.mat")


def matrix_to_position(mat):
    x_min = min(mat[0][0], mat[3][0])
    x_max = max(mat[1][0], mat[2][0])
    y_min = min(mat[0][1], mat[1][1])
    y_max = max(mat[2][1], mat[3][1])
    return x_min, y_min, x_max, y_max


image_folder = "J:/multi"
save_folder = "J:/bb"
gallery_folder = "J:/gallery"
prefix = "AcquisitionMultipleCamera"
camera_id = {"l": "18274603", "m": "18274611", "r": "18274488"}
total_frame = 29
total_id = 16

for view in ["l", "m", "r"]:
    for frame in range(total_frame):
        image_path = os.path.join(image_folder, prefix + "-" + camera_id[view] + "-" + str(frame) + ".bmp")
        image = Image.open(image_path)

        if view == "r":
            image = image.rotate(-17)
        if view == "l":
            image = image.rotate(4)

        draw = ImageDraw.Draw(image)
        gallery_image = image.copy()

        for id in range(total_id):
            matrix = m["Ax_"+view][frame][id]
            if matrix.shape == (4, 2):
                x_min, y_min, x_max, y_max = matrix_to_position(matrix)
                draw.line((x_min, y_min, x_max, y_min), fill="red", width=5)
                draw.line((x_max, y_min, x_max, y_max), fill="red", width=5)
                draw.line((x_max, y_max, x_min, y_max), fill="red", width=5)
                draw.line((x_min, y_max, x_min, y_min), fill="red", width=5)

                crop_image = gallery_image.crop((x_min, y_min, x_max, y_max))
                gallery_path = os.path.join(gallery_folder, view, str(frame))
                if not os.path.isdir(gallery_path):
                    os.mkdir(gallery_path)
                crop_image.save(os.path.join(gallery_path, str(id) + ".bmp"))

        image.save(os.path.join(save_folder, view, str(frame) + ".bmp"))
