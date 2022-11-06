import os.path

import cv2
import numpy as np
import pandas as pd

dir_test_list_csv = '../dataset/testA_new/test_list.csv'
dir_pred = './pred'
dir_export = './masks'

if __name__ == '__main__':
    border_list = pd.read_csv(dir_test_list_csv)

    if not os.path.exists(dir_export):
        os.makedirs(dir_export)

    for i in border_list.index:
        img_name = border_list.iloc[i, 0]
        img_name = f'{img_name.split(".")[0]}.png'
        border = border_list.iloc[i, 1:].values.astype(int)
        x0, y0 = border[0], border[1]
        x1, y1 = border[2], border[3]
        x_max, y_max = border[4], border[5]

        mask = np.zeros((y_max, x_max))
        fill = cv2.imread(os.path.join(dir_pred, img_name), cv2.IMREAD_GRAYSCALE)
        mask[y0:y1, x0:x1] = fill
        cv2.imwrite(os.path.join(dir_export, img_name), mask)
