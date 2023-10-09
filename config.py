import torch
import random
import numpy as np

# Set seed
SEED = 42
fold_num = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data_config = {
    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': "/nvmedata/fengxingyu/preprocessed",
    'crop_size': (96, 96, 96),
    'overlap_size' : (16, 16, 16),
    'spacing': (1.0, 1.0, 1.0),
    'pad_value': 170,
    "stride": 4,
    'rcnn_size': [7, 7, 7],
    'bound_size': 12,
}

net_config = {
    # Net configuration
    'chanel': 1,

    # loss setting
    'loss_weight_dict': {'cls_weight': 4.0, "shape_weight": 0.1, "offset_weight": 1.0, "iou_weight": 1.0},
    'topk': 7,
}

train_config = {
    'batch_size': 4,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'warm_up': 30,

    'epochs': 100,
    'epoch_save': 5,
    'num_workers': 10,

    'out_dir': f'experiment/fold{fold_num}',
    'save_dir': f'experiment/fold{fold_num}',

    'data_dir' : data_config['preprocessed_data_dir'],

    'fold_setting': f"splits/split_fold{fold_num}.json",
    'ckpt': f"experiment/fold{fold_num}/detector100.ckpt",
}

config = dict(data_config, **net_config)
config = dict(config, **train_config)
