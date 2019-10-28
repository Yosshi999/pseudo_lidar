from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import logger
from dataloader import KITTILoader3D as ls
from dataloader import KITTILoader_dataset3d as DA
from models import *
from models.V2_util import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr_scale', type=int, default=200, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='Kitti/object/train.txt',
                    help='save model')
parser.add_argument('--btrain', type=int, default=4)
parser.add_argument('--start_epoch', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.savemodel):
    os.makedirs(args.savemodel)
print(os.path.join(args.savemodel, 'training.log'))
log = logger.setup_logger(os.path.join(args.savemodel, 'training.log'))
import datetime
log.info(datetime.datetime.now())

all_left_img, all_right_img, all_left_disp, all_calib, = ls.dataloader(args.datapath,
                                                            args.split_file, True)
all_left_img_v, all_right_img_v, all_left_disp_v, all_calib_v, = ls.dataloader(args.datapath,
                                                            args.split_file.replace('train', 'val'), True)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_calib, True),
    batch_size=args.btrain, shuffle=True, num_workers=14, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img_v, all_right_img_v, all_left_disp_v, all_calib_v, False),
    batch_size=args.btrain, shuffle=False, num_workers=14, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    log.info('load model ' + args.loadmodel)
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L, calib):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))
    calib = Variable(torch.FloatTensor(calib))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        calib = calib.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR, calib)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        depth_true = disp2depth_torch(calib.view(-1, 1, 1), disp_true)[mask]
        loss = 0.5 * F.smooth_l1_loss(output1[mask], depth_true, size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], depth_true, size_average=True) + F.smooth_l1_loss(output3[mask], depth_true,
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR, calib)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp2depth_torch(calib, disp_true[mask]), size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data.item()


def test(imgL, imgR, disp_true, calib):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    calib = Variable(torch.FloatTensor(calib))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
        calib = calib.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR, calib)

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    depth_true = disp2depth_torch(calib, disp_true)
    true_depth = depth_true
    index = np.argwhere(true_depth > 0)
    depth_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_depth[index[0][:], index[1][:], index[2][:]] - pred_depth[index[0][:], index[1][:], index[2][:]])
    correct = (depth_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            depth_true[index[0][:], index[1][:], index[2][:]] < true_depth[
        index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.lr_scale:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_test_loss= 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, calib) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L, calib)
            log.info('Iter %d training loss = %.3f , time = %.2f , progress = %.2f' % (batch_idx, loss, time.time() - start_time, (batch_idx+1.0)/len(TrainImgLoader)))
            total_train_loss += loss
        log.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

        ## test ##
        for batch_idx, (imgL, imgR, disp_L, calib) in enumerate(TestImgLoader):
            test_loss = test(imgL,imgR, disp_L, calib)
            log.info('Iter %d 3-px error in val = %.3f , progress = %.2f' %(batch_idx, test_loss*100, (batch_idx+1.0)/len(TestImgLoader)))
            total_test_loss += test_loss


        log.info('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
        if total_test_loss/len(TestImgLoader)*100 > max_acc:
            max_acc = total_test_loss/len(TestImgLoader)*100
            max_epo = epoch
        log.info('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))

        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
