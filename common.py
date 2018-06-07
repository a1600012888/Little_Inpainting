import torch
import os
import json


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = 'L1+style+content+tv'
    data_dir = '/unsullied/sharefs/zhangtianyuan/sfzhang/cvdl/cityscapes/data'
    exp_dir = os.path.join('/unsullied/sharefs/zhangtianyuan/sfzhang/cvdl_inpainting/', exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')

    def make_dir(self):
        self.exp_dir = os.path.join('/unsullied/sharefs/zhangtianyuan/sfzhang/cvdl_inpainting/', self.exp_name)
        if os.path.exists(self.exp_dir) == False:
            os.makedirs(os.path.join(self.exp_dir, 'model'))
            os.makedirs(os.path.join(self.exp_dir, 'log'))
        self.log_dir = os.path.join(self.exp_dir, 'log/')
        self.model_dir = os.path.join(self.exp_dir, 'model/')

config = Config()
