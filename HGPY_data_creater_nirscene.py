import numpy as np
import cv2 as cv
import os

in_path_base = 'nirscene1'
out_path_base = 'data/nirscene1'
in_path_sub = ['country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water']

idx = 0

for path_sub in in_path_sub:
    in_path = in_path_base + '/' + path_sub
    img_list = os.listdir(in_path)
    img_list.sort()
    print(in_path)
    for i in range(len(img_list) // 2):
        print(i)
        img_path_nir = in_path + "/" + img_list[2 * i]
        img_path_rgb = in_path + "/" + img_list[2 * i + 1]
        img_nir = cv.imread(img_path_nir, 1)
        img_rgb = cv.imread(img_path_rgb, 1)
        m, n, _ = img_nir.shape
        m //= 64
        n //= 64
        for a in range(m):
            for b in range(n):
                out_path_nir = out_path_base + "/{:0>6d}_nir.png".format(idx)
                out_path_rgb = out_path_base + "/{:0>6d}_rgb.png".format(idx)
                tmp_nir = img_nir[a * 64:(a + 1) * 64, b * 64:(b + 1) * 64, :]
                tmp_rgb = img_rgb[a * 64:(a + 1) * 64, b * 64:(b + 1) * 64, :]
                cv.imwrite(out_path_nir, tmp_nir)
                cv.imwrite(out_path_rgb, tmp_rgb)
                idx += 1
