import torch
import os
import json


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = 'L1+style'
    data_dir = '/data1/wurundi/cityscapes/data'
    exp_dir = os.path.join('/data1/wurundi/cityscapes/', exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')

    if os.path.exists(exp_dir) == False:
        os.makedirs(os.path.join(exp_dir, 'model'))
        os.makedirs(os.path.join(exp_dir, 'log'))


config = Config()