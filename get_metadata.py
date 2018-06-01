import os
import cv2 as cv
import json
import numpy as np
ORI_DATA_ROOT = '/data1/wurundi/cityscapes/leftImg8bit_trainvaltest/leftImg8bit'
NEW_DATA_ROOT = '/data1/wurundi/cityscapes/data'

def process_data(phase):
    results = []
    data_root = os.path.join(NEW_DATA_ROOT, phase)
    if os.path.exists(data_root) == False:
        os.mkdir(data_root)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ORI_DATA_ROOT, phase)):
        for filepath in filenames:
            if filepath[0] == '.':
                continue
            ori_img = cv.imread(os.path.join(dirpath, filepath))
            ori_img = cv.resize(ori_img, (512, 256))
            img_path = os.path.join(data_root, filepath)
            cv.imwrite(img_path, ori_img)

            sy = np.random.randint(0, 255-64)
            sx = np.random.randint(0, 511-128)
            ey = sy + 64
            ex = sx + 128
            results.append({'img_path': img_path,
                            'sy': sy,
                            'sx': sx,
                            'ey': ey,
                            'ex': ex
                            })
    with open(os.path.join(NEW_DATA_ROOT, phase+'.json'), 'w') as fp:
        json.dump(results, fp)
    print('processing data is done.')

if __name__ == '__main__':
    process_data('train')
    process_data('val')
    process_data('test')