from datetime import datetime
import os, sys
import os.path as osp
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from od_oc_segmentation.train_process import Trainer

from od_oc_segmentation.dataloaders import fundus_dataloader as DL
from od_oc_segmentation.dataloaders import custom_transforms as tr
from od_oc_segmentation.networks.baseline import deeplabv3plus_mobilenet
from od_oc_segmentation.test import main_test
from datetime import datetime
import pytz
from torch import nn
from od_oc_segmentation.utils.Utils import *

local_path = osp.dirname(osp.abspath(__file__))


def main_train(args):
    now = datetime.now()
    args.out = osp.join(local_path, 'logs', 'test'+str(args.datasetTest[0]), 'lam'+str(args.lam), now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    if len(str2list(args.gpu))>1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda:0")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                                         transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                       transform=composed_transforms_ts)
    val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. model
    model = deeplabv3plus_mobilenet(args.atom, num_classes=2, output_stride=args.out_stride,
                                    pretrained_backbone=True)
    print('parameter numer:', sum([p.numel() for p in model.parameters()]))
    if len(str2list(args.gpu))>1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model = model.cuda()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    trainer = Trainer.Trainer(
        cuda=cuda,
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        gpu=args.gpu
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

    return args.out


def write_mean_std(meter, datasetTest, statistic='mean'):
    if statistic == 'mean':
        val_cup_dice = meter.mean[0]
        val_disc_dice = meter.mean[1]
        total_hd_OC = meter.mean[2]
        total_hd_OD = meter.mean[3]
        total_asd_OC = meter.mean[4]
        total_asd_OD = meter.mean[5]
    else:
        val_cup_dice = meter.std[0]
        val_disc_dice = meter.std[1]
        total_hd_OC = meter.std[2]
        total_hd_OD = meter.std[3]
        total_asd_OC = meter.std[4]
        total_asd_OD = meter.std[5]

    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))

    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
    print('''\n==>average_hd_OC : {0}'''.format(total_hd_OC))
    print('''\n==>average_hd_OD : {0}'''.format(total_hd_OD))
    print('''\n==>ave_asd_OC : {0}'''.format(total_asd_OC))
    print('''\n==>average_asd_OD : {0}'''.format(total_asd_OD))

    test_prediction_save_path = './results/'
    output_path = os.path.join(test_prediction_save_path, 'test' + str(datasetTest))

    with open(osp.join(output_path, './test' + str(datasetTest) + '_log.csv'), 'a') as f:
        log = [[statistic+'-batch-size: '] + [args.batch_size] + [model_file] + [' '] + \
               [val_cup_dice] + [' '] + \
               [val_disc_dice] + [' '] + \
               [total_hd_OC] + [' '] + \
               [total_hd_OD] + [' '] + \
               [total_asd_OC] + [' '] + \
               [total_asd_OD] + [' ']]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-r', '--runs', type=int, default=10)
    parser.add_argument('-g', "--gpu", type=str, default='0', help='visible gpu id')

    parser.add_argument('--datasetTrain', nargs='+', type=int, default=1, help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=1, help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=60, help='max epoch')
    parser.add_argument('-s', '--stop-epoch', type=int, default=40, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=1, help='interval epoch number to valide the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='./data/Fundus/', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)

    parser.add_argument('-a', "--atom", type=str, default='TVConv', choices=['TVConv', 'base', 'DW'])

    args = parser.parse_args()

    meter = AverageValueMeter()
    for i in range(0, args.runs):
        model_file = main_train(args)
        val_cup_dice, val_disc_dice, total_hd_OC, total_hd_OD, total_asd_OC, total_asd_OD = main_test(
            model_file+'/checkpoint_best.pth.tar', args)
        meter.add(np.array([val_cup_dice, val_disc_dice, total_hd_OC, total_hd_OD, total_asd_OC, total_asd_OD]))

    write_mean_std(meter, args.datasetTest[0], statistic='mean')
    write_mean_std(meter, args.datasetTest[0], statistic='std')