import os
import sys
import torch
import traceback
import argparse
import setproctitle

from utils.misc import Logger
from config import train_config, config
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torch.utils.tensorboard import SummaryWriter
from models.centernet import build_detector
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from monai.data import DataLoader, load_decathlon_datalist
from monai.data.utils import no_collation
from datasets.bbox_reader import Luna16
from datasets.transforms import generate_train_transform, generate_val_transform
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

this_module = sys.modules[__name__]

parser = argparse.ArgumentParser(description='PyTorch Detector')

parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--ckpt', default=None, type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--init-lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument("--fold-setting", type=str, default=config['fold_setting'],
                    help="path to save the results")
parser.add_argument('--data-dir', default=train_config['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--save-dir', default=train_config['save_dir'], type=str, metavar='OUT',
                    help='path to save data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--warm-up', default=train_config['warm_up'], type=int, metavar='OUT',
                    help='epochs for warm up')

## global part
scaler = torch.cuda.amp.GradScaler()
args = parser.parse_args()

loss_dict = config['loss_weight_dict']
setproctitle.setproctitle("CPMV2-Fold0-Train")

def main():
    # Load training configuration
    cudnn.benchmark = True
    initial_checkpoint = args.ckpt

    train_data = load_decathlon_datalist(
        args.fold_setting,
        is_segmentation=True,
        data_list_key="training",
        base_dir=args.data_dir,
    )

    train_transform = generate_train_transform(config)
    val_transform = generate_val_transform()

    train_dataset = Luna16(data=train_data[: int(
        0.95 * len(train_data))], transform=train_transform)

    val_ds = Luna16(
        data=train_data[int(0.95 * len(train_data)):],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=no_collation,
        persistent_workers=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=7,
        pin_memory=True,
        collate_fn=no_collation,
        persistent_workers=True)

    # Initilize network
    detector = build_detector(config, device).to(device)

    optimizer = torch.optim.SGD(detector.parameters(
    ), lr=args.init_lr, momentum=args.momentum, nesterov=True)
    lr_warmup = LinearLR(optimizer, start_factor=0.01,
                         end_factor=1, total_iters=args.warm_up)
    cos_lr = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    swa_model = torch.optim.swa_utils.AveragedModel(detector)
    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        # state = net.state_dict()
        # state.update(checkpoint['state_dict'])
        try:
            detector.load_state_dict(checkpoint['state_dict'])
        except:
            print('Load something failed!')
            traceback.print_exc()

    start_epoch = start_epoch + 1
    tb_out_dir = os.path.join(args.out_dir, 'runs')
    logfile = os.path.join(args.out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))

    for epoch in range(start_epoch, args.epochs + 1):
        # learning rate schedule
        train(detector, train_loader, optimizer, epoch, train_writer)
        validate(detector, val_loader, epoch, val_writer)

        if epoch > args.epochs * 0.90:
            swa_model.update_parameters(detector)
            swa_state_dict = swa_model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'state_dict': swa_state_dict,
                'args': args},
                os.path.join(args.save_dir, f"swa_detector_{epoch}.ckpt"))

        if epoch < args.warm_up:
            lr_warmup.step()
        else:
            cos_lr.step()

        if epoch % 5 == 0:
            state_dict = detector.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'args': args},
                os.path.join(args.save_dir, f"detector_{epoch}.ckpt"))

    writer.close()
    train_writer.close()
    val_writer.close()


def train(detector, train_loader, optimizer, epoch, writer):
    detector.train()

    current_lr = optimizer.param_groups[-1]['lr']

    total_cls_loss = 0
    total_shape_loss = 0
    total_offset_loss = 0
    total_iou_loss = 0

    iter = 0

    for batch_data in tqdm(train_loader):
        iter += 1
        inputs = torch.stack([batch_data_ii["image"].to(
            device) for batch_data_i in batch_data for batch_data_ii in batch_data_i])

        targets = [
            dict(
                label=batch_data_ii["label"].to(device),
                box=batch_data_ii["box"].to(device),
            )
            for batch_data_i in batch_data
            for batch_data_ii in batch_data_i
        ]

        cls_loss, shape_loss, offset_loss, iou_loss = detector(
            [inputs, targets])
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(
        ), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()

        loss = cls_loss * loss_dict['cls_weight'] + shape_loss * loss_dict['shape_weight'] + offset_loss * loss_dict['offset_weight'] + iou_loss * loss_dict['iou_weight']

        total_cls_loss += loss_dict['cls_weight'] * cls_loss.item()
        total_shape_loss += loss_dict['shape_weight'] * shape_loss.item()
        total_offset_loss += loss_dict['offset_weight'] * offset_loss.item()
        total_iou_loss += loss_dict['iou_weight'] * iou_loss.item()

        optimizer.zero_grad()
        # need to backward loss
        loss.backward()
        optimizer.step()

    writer.add_scalars("Train part", {
        'train_cls_loss': total_cls_loss / iter,
        'train_shape_loss': total_shape_loss / iter,
        'train_offset_loss': total_offset_loss / iter,
        'train_iou_loss': total_iou_loss / iter,
    }, epoch)

    writer.add_scalar("lr", current_lr, epoch)

def validate(detector, val_loader, epoch, writer):
    detector.train()

    total_cls_loss = 0
    total_shape_loss = 0
    total_offset_loss = 0
    total_iou_loss = 0

    iter = 0

    for batch_data in tqdm(val_loader):
        with torch.no_grad():
            iter += 1
            inputs = torch.stack([batch_data_ii["image"].to(
                device) for batch_data_i in batch_data for batch_data_ii in batch_data_i])

            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            cls_loss, shape_loss, offset_loss, iou_loss = detector(
                [inputs, targets])
            cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(
            ), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()

            total_cls_loss += loss_dict['cls_weight'] * cls_loss.item()
            total_shape_loss += loss_dict['shape_weight'] * shape_loss.item()
            total_offset_loss += loss_dict['offset_weight'] * \
                offset_loss.item()
            total_iou_loss += loss_dict['iou_weight'] * iou_loss.item()

    writer.add_scalars("Val part", {
        'val_cls_loss': total_cls_loss / iter,
        'val_shape_loss': total_shape_loss / iter,
        'val_offset_loss': total_offset_loss / iter,
        'val_iou_loss': total_iou_loss / iter,
    }, epoch)

if __name__ == '__main__':
    main()
