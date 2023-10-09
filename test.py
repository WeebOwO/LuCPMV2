import os
import sys
import time
import math
import torch

import argparse
import logging
import warnings
import numpy as np
import pandas as pd

from config import config
from datasets.bbox_reader import Luna16
from datasets.transforms import generate_infer_transform
from datasets.split_combine import SplitComb

from utils.misc import Logger
from utils.box_utils import nms_3D

from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
from models.backbone import Detection_Postprocess
from models.centernet import build_detector
from monai.data import load_decathlon_datalist

this_module = sys.modules[__name__]
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default="eval",
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['ckpt'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--fold-setting", type=str, default=config['fold_setting'],
                    help="path to save the results")

detection_postprocess = Detection_Postprocess(topk=60, threshold=0.8, nms_threshold=0.05, num_topk=20, crop_size=config['crop_size'])

def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    load_swa = False

    if args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        test_set_name = args.fold_setting
        initial_checkpoint = args.weight
        net = build_detector(config, device)
        swa_model = torch.optim.swa_utils.AveragedModel(net)
        out_dir = args.out_dir
        
        if initial_checkpoint:
            if load_swa:
                print('-- Loading model from {}...'.format(initial_checkpoint))
                checkpoint = torch.load(initial_checkpoint)
                epoch = checkpoint['epoch']
                swa_model.load_state_dict(checkpoint['state_dict'])
            else:
                print('-- Loading model from {}...'.format(initial_checkpoint))
                checkpoint = torch.load(initial_checkpoint)
                epoch = checkpoint['epoch']
                net.load_state_dict(checkpoint['state_dict'])
        else:
            print('-- No model weight file specified. ✘')
            return

        detector = swa_model if load_swa else net
        detector.to(device)
        save_dir = os.path.join(out_dir, 'res', 'val' + str(epoch))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))

        logfile = os.path.join(out_dir, 'log_test.txt')
        sys.stdout = Logger(logfile)

        inference_data = load_decathlon_datalist(
            test_set_name,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=data_dir,  
        )
        
        id_list = []
        for data in inference_data:
            filename = data['image'].split('/')[-1][:-4]
            id_list.append(filename)
            
        infer_transform = generate_infer_transform()
        dataset = Luna16(data=inference_data, transform=infer_transform, mode='eval')

        eval(detector, dataset, save_dir, id_list)
    else:
        logging.error('-- Mode %s is not supported. ✘' % (args.mode))

def eval(detector, dataset, save_dir=None, id_list=None):
    split_comber = SplitComb(crop_size=config['crop_size'], overlap=config['overlap_size'], pad_value=-1)
    detector.eval()

    print('Total # of eval data {}'.format(len(dataset)))
    time_start = time.perf_counter()
    # hard code for now, may be change in the future
 
    top_k = 40
    split_batch_size = 8

    for i, targets in enumerate(dataset):
        image = targets['image'].squeeze()
        split_images, nzhw = split_comber.split(image)
        data = torch.from_numpy(split_images)
        pid = id_list[i]
        outputlist = []

        print('-- Scan #{} pid:{} \n-- Predicting {}...'.format(i, pid, image.shape))
            
        for j in range(int(math.ceil(data.size(0)) / split_batch_size)):
            end = (j + 1) * split_batch_size
            input = data[j * split_batch_size : end].to(device)

            with torch.no_grad():
                output = detector(input)
                output = detection_postprocess(output, device=device)

            outputlist.append(output.data.cpu().numpy())
        
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8)
        object_ids = output[:, 0] != -1.0
        output = output[object_ids]

        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=0.05, top_k=top_k)
            output = output[keep]

    time_end = time.perf_counter()
    time_last = time_end - time_start

    print(f"Total time is {time_last}")
    print(f"Time / scan is {time_last / len(dataset)}")

    test_res = []
    for pid in id_list:
        if os.path.exists(os.path.join(save_dir, '%s_pbb.npy' % (pid))):
            bboxs = np.load(os.path.join(save_dir, '%s_pbb.npy' % (pid)))
            bboxs = bboxs[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(bboxs))
            test_res.append(np.concatenate([names, bboxs], axis=1))

    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')

    test_res = np.concatenate(test_res, axis = 0)
    test_submission_path = os.path.join(eval_dir, 'test_result.csv')
    df = pd.DataFrame(test_res, columns=col_names)
    df.to_csv(test_submission_path, index=False)

    if not os.path.exists(os.path.join(eval_dir, 'test_res')):
        os.makedirs(os.path.join(eval_dir, 'test_res'))

    noduleCADEvaluation('annos/voxel_annotations.csv',
                        'annos/voxel_annotations_exclude.csv',
                        id_list, test_submission_path, os.path.join(eval_dir, 'test_res'))
    
if __name__ == '__main__':
    main()
