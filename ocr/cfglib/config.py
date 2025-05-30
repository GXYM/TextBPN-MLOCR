from easydict import EasyDict
import torch
import os

config = EasyDict()


# Normalize image
config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

config.gpu = "0"

# Experiment name #
config.exp_name = "pre-training"

# dataloader jobs number
config.num_workers = 8

# batch_size
config.batch_size = 12

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.local_rank = 0

config.use_amp = True

config.accum_grad_iters = 1


config.output_dir = 'output'

config.input_size = 640

# max polygon per image
# synText, total-text:64; CTW1500: 64; icdar: 64;  MLT: 32; TD500: 64.
config.max_annotation = 512

# adj num for graph
config.adj_num = 4

# control points number
config.num_points = 48

# use hard examples (annotated as '#')
config.use_hard = True

# Load data into memory at one time
config.load_memory = False

# # clip gradient of loss
config.grad_clip = 10.0

# prediction on 1/scale feature map
config.scale = 1

# demo tcl threshold
config.dis_threshold = 0.3

config.cls_threshold = 0.6

config.test_size= [0, 1024] 


# Contour approximation factor
config.approx_factor = 0.004


# os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
# 设置默认的数据类型为 float32
torch.set_default_dtype(torch.float32)
config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
