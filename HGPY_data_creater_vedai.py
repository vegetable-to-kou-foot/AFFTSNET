import numpy as np
import cv2 as cv
import os

in_path_base = 'nirscene1/'
out_path_base = 'data/nirscene1'

idx = 0

for i in range(689):
    if i % 10 == 0:
        print(i)
    in_path_1 = in_path_base + "/{:0>8d}_co.png".format(i)
    in_path_2 = in_path_base + "/{:0>8d}_ir.png".format(i)

    if os.path.isfile(in_path_1) and os.path.isfile(in_path_2):
        img_1 = cv.imread(in_path_1)
        img_2 = cv.imread(in_path_2)
        for j in range(4):
            out_path_1 = out_path_base + "/{:0>8d}_co.png".format(idx)
            out_path_2 = out_path_base + "/{:0>8d}_ir.png".format(idx)
            tmp_1 = img_1[j * 128:(j + 1) * 128, j * 128:(j + 1) * 128]
            tmp_2 = img_2[j * 128:(j + 1) * 128, j * 128:(j + 1) * 128]
            cv.imwrite(out_path_1, tmp_1)
            cv.imwrite(out_path_2, tmp_2)
            idx = idx + 1
