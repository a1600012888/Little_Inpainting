import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
from model import network, vgg_for_style_transfer
from common import config
from utils import TrainClock, save_args, AverageMeter, write_avgs, write_tensor
from dataset import get_dataloaders
from loss_function import gram_matrix

perceptual_content_name = ['Per/Train/cont1', 'Per/Train/cont2', 'Per/Train/cont3', 'Per/Train/cont4', 'Per/Train/cont5']
perceptual_style_name = ['PerTrain/style1', 'PerTrain/style2', 'PerTrain/style3', 'PerTrain/style4', 'PerTrain/style5']
t_perceptual_content_name = ['Per/Val/cont1', 'Per/Val/cont2', 'Per/Val/cont3', 'Per/Val/cont4', 'Per/Val/cont5']
t_perceptual_style_name = ['PerVal/style1', 'PerVal/style2', 'PerVal/style3', 'PerVal/style4', 'PerVal/style5']

torch.backends.cudnn.benchmark = True

Valid_Loss_weight = 1
Hole_Loss_weight = 6
Content_Loss_weight = 0.01
Style_Loss_weight = 30
Tv_Loss_weight = 0.1

rValid_Loss_weight = 1
rHole_Loss_weight = 6
rContent_Loss_weight = 0.01
rStyle_Loss_weight = 30
rTv_Loss_weight = 0.1

class Session:

    def __init__(self, config, net=None):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.best_val_loss = np.inf
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.clock = TrainClock()

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            'state_dict': self.net.state_dict(),
            'best_val_loss': self.best_val_loss,
            'clock': self.clock.make_checkpoint(),
        }
        torch.save(tmp, ckp_path)

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.best_val_loss = checkpoint['best_val_loss']


def train_model(train_loader, model, vgg, criterion, optimizer, epoch, tb_writer):
    losses = AverageMeter()
    hole_losses = AverageMeter()
    valid_losses = AverageMeter()
    style_losses = AverageMeter()
    content_losses = AverageMeter()
    tv_losses = AverageMeter()

    s1 = AverageMeter()
    s2 = AverageMeter()
    s3 = AverageMeter()
    s4 = AverageMeter()
    s5 = AverageMeter()
    # ensure model is in train mode

    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs = data['hole_img'].float()
        labels = data['ori_img'].float()
        ori_img = labels.clone()
        # mask: 1 for the hole and 0 for others
        masks = data['mask'].float()
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        masks = masks.to(config.device)
        ori_img = ori_img.to(config.device)

        # pass this batch through our model and get y_pred
        outputs = model(inputs)
        # use five different level features, each are extracted after down-sampling
        targets = vgg(ori_img)
        features = vgg(outputs)

        # get content and style loss
        content_loss = 0
        style_loss = 0

        now_style_loss = [0.0, 0.0, 0.0, 0.0, 0.0]#np.ndarray(shape=(5, ))
        for k in range(inputs.size(0)):
            content_loss += torch.sum((features[3][k] - targets[3][k]) ** 2) / 2
            #now_content_loss = F.mse_loss(features[3][k], targets[3][k])
            #content_loss = content_loss + now_content_loss
            targets_gram = [gram_matrix(f[k]) for f in targets]
            features_gram = [gram_matrix(f[k]) for f in features]


            #style_loss += torch.sum(torch.mean((targets - features_gram) ** 2, dim = 0))
            for j in range(len(targets_gram)):
                now_style_loss[j] = torch.sum((features_gram[j] - targets_gram[j]) ** 2)
                style_loss = style_loss + now_style_loss[j]
        style_loss /= inputs.size(0)
        content_loss /= inputs.size(0)
        style_losses.update(style_loss.item(), inputs.size(0))
        content_losses.update(content_loss.item(), inputs.size(0))

        # update loss metric
        # suppose criterion is L1 loss
        hole_loss = criterion(outputs*masks, labels*masks)
        valid_loss = criterion(outputs*(1-masks), labels*(1-masks))
        hole_losses.update(hole_loss.item(), inputs.size(0))
        valid_losses.update(valid_loss.item(), inputs.size(0))


        write_avgs([s1, s2, s3, s4, s5], now_style_loss)

        # get total variation loss
        outputs_hole = outputs * masks
        targets_hole = labels * masks
        tv_loss = torch.sum(torch.abs(outputs_hole[:, :, :, 1:] - targets_hole[:, :, :, :-1])) \
                  + torch.sum(torch.abs(outputs_hole[:, :, 1:, :] - targets_hole[:, :, :-1, :]))
        tv_loss /= inputs.size(0)
        tv_losses.update(tv_loss.item(), inputs.size(0))

        # total loss
        loss = hole_loss * rHole_Loss_weight + valid_loss * rValid_Loss_weight + \
               style_loss * rStyle_Loss_weight + content_loss * rContent_Loss_weight + \
               tv_loss * rTv_Loss_weight
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            loss="LOSS:{:.4f}".format(losses.avg))

    tb_writer.add_scalar('train/epoch_loss', losses.avg, epoch)
    tb_writer.add_scalar('train/hole_loss', hole_losses.avg * Hole_Loss_weight, epoch)
    tb_writer.add_scalar('train/valid_loss', valid_losses.avg * Valid_Loss_weight, epoch)
    tb_writer.add_scalar('train/style_loss', style_losses.avg * Style_Loss_weight, epoch)
    tb_writer.add_scalar('train/content_loss', content_losses.avg * Content_Loss_weight, epoch)
    tb_writer.add_scalar('train/tv_loss', tv_losses.avg * Tv_Loss_weight, epoch)

    write_tensor(perceptual_style_name, [s1, s2, s3, s4, s5], epoch, tb_writer)

    torch.cuda.empty_cache()
    return


def valid_model(valid_loader, model, vgg, criterion, optimizer, epoch, tb_writer):
    losses = AverageMeter()
    hole_losses = AverageMeter()
    valid_losses = AverageMeter()
    style_losses = AverageMeter()
    content_losses = AverageMeter()
    tv_losses = AverageMeter()

    s1 = AverageMeter()
    s2 = AverageMeter()
    s3 = AverageMeter()
    s4 = AverageMeter()
    s5 = AverageMeter()
    # ensure model is in train mode
    model.eval()
    vgg.eval()
    pbar = tqdm(valid_loader)
    for i, data in enumerate(pbar):
        inputs = data['hole_img'].float()
        labels = data['ori_img'].float()
        ori_img = labels.clone()
        # mask: 1 for the hole and 0 for others
        masks = data['mask'].float()
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        masks = masks.to(config.device)
        ori_img = ori_img.to(config.device)

        with torch.no_grad():
            # pass this batch through our model and get y_pred
            outputs = model(inputs)
            targets = vgg(ori_img)
            features = vgg(outputs)

            # get content and style loss
            content_loss = 0
            style_loss = 0

            now_style_loss = [0.0, 0.0, 0.0, 0.0, 0.0]  # np.ndarray(shape=(5, ))
            for k in range(inputs.size(0)):
                content_loss += torch.sum((features[3][k] - targets[3][k]) ** 2) / 2
                #now_content_loss = F.mse_loss(features[3][k], targets[3][k])
                #content_loss = content_loss + now_content_loss
                targets_gram = [gram_matrix(f[k]) for f in targets]
                features_gram = [gram_matrix(f[k]) for f in features]

                # style_loss += torch.sum(torch.mean((targets - features_gram) ** 2, dim = 0))
                for j in range(len(targets_gram)):
                    now_style_loss[j] = torch.sum((features_gram[j] - targets_gram[j]) ** 2)
                    style_loss = style_loss + now_style_loss[j]

            style_loss /= inputs.size(0)
            content_loss /= inputs.size(0)
            style_losses.update(style_loss.item(), inputs.size(0))
            content_losses.update(content_loss.item(), inputs.size(0))

            # update loss metric
            # suppose criterion is L1 loss
            hole_loss = criterion(outputs*masks, labels*masks)
            valid_loss = criterion(outputs*(1-masks), labels*(1-masks))
            hole_losses.update(hole_loss.item(), inputs.size(0))
            valid_losses.update(valid_loss.item(), inputs.size(0))

            # get total variation loss
            outputs_hole = outputs * masks
            targets_hole = labels * masks
            tv_loss = torch.sum(torch.abs(outputs_hole[:, :, :, 1:] - targets_hole[:, :, :, :-1])) \
                      + torch.sum(torch.abs(outputs_hole[:, :, 1:, :] - targets_hole[:, :, :-1, :]))
            tv_loss /= inputs.size(0)
            tv_losses.update(tv_loss.item(), inputs.size(0))

            # total loss
            loss = hole_loss * rHole_Loss_weight + valid_loss  * rValid_Loss_weight+ \
                   style_loss * rStyle_Loss_weight + content_loss * rContent_Loss_weight+ \
                   tv_loss * rTv_Loss_weight
            losses.update(loss.item(), inputs.size(0))

            write_avgs([s1, s2, s3, s4, s5], now_style_loss)
            if i == 0:
                for j in range(min(inputs.size(0), 3)):
                    hole_img = data['hole_img'][j]
                    ori_img = data['ori_img'][j]
                    out_img = outputs[j].detach()
                    out_img = out_img / (torch.max(out_img) - torch.min(out_img))
                    tb_writer.add_image('valid/ori_img{}'.format(j), ori_img, epoch)
                    tb_writer.add_image('valid/hole_img{}'.format(j), hole_img, epoch)
                    tb_writer.add_image('valid/out_img{}'.format(j), out_img, epoch)

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(valid_loader)))
        pbar.set_postfix(
            loss="LOSS:{:.4f}".format(losses.avg))

    tb_writer.add_scalar('valid/epoch_loss', losses.avg, epoch)
    tb_writer.add_scalar('valid/hole_loss', hole_losses.avg * Hole_Loss_weight, epoch)
    tb_writer.add_scalar('valid/valid_loss', valid_losses.avg  * Valid_Loss_weight, epoch)
    tb_writer.add_scalar('valid/style_loss', style_losses.avg * Style_Loss_weight, epoch)
    tb_writer.add_scalar('valid/content_loss', content_losses.avg * Content_Loss_weight, epoch)
    tb_writer.add_scalar('valid/tv_loss', tv_losses.avg * Tv_Loss_weight, epoch)

    write_tensor(t_perceptual_style_name, [s1, s2, s3, s4, s5], epoch, tb_writer)

    torch.cuda.empty_cache()
    outspects = {
        'epoch_loss': losses.avg,
    }
    return outspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int, help='epoch number')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
    parser.add_argument('-b', '--batch_size', default=6, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('--exp_name', default=config.exp_name, type=str, required=False)
    parser.add_argument('--enable_l1', action = 'store_true')
    parser.add_argument('--tv', type = float, default = 0.1)
    parser.add_argument('--style', type = float, default = 30)
    args = parser.parse_args()
    print(args)
    global rValid_Loss_weight
    global rHole_Loss_weight
    global rStyle_Loss_weight
    global rTv_Loss_weight
    global rContent_Loss_weight

    rStyle_Loss_weight = args.style
    rTv_Loss_weight = args.tv
    if not args.enable_l1:
        rValid_Loss_weight = 0
        rHole_Loss_weight = 0
    config.exp_name = args.exp_name
    config.make_dir()

    save_args(args, config.log_dir)
    net = network()
    vgg = vgg_for_style_transfer()

    net = torch.nn.DataParallel(net).cuda()
    vgg = torch.nn.DataParallel(vgg).cuda()

    sess = Session(config, net=net)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    train_loader = get_dataloaders(os.path.join(config.data_dir, 'train.json'),
                                   batch_size=args.batch_size, shuffle=True)
    valid_loader = get_dataloaders(os.path.join(config.data_dir, 'val.json'),
                                   batch_size=args.batch_size, shuffle=True)

    clock = sess.clock
    tb_writer = sess.tb_writer

    criterion = nn.L1Loss().cuda()

    optimizer = optim.Adam(sess.net.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True)

    for e in range(args.epochs):
        train_model(train_loader, sess.net, vgg,
                                criterion, optimizer, clock.epoch, tb_writer)
        valid_out = valid_model(valid_loader, sess.net, vgg,
                                criterion, optimizer, clock.epoch, tb_writer)

        tb_writer.add_scalar('train/learning_rate', optimizer.param_groups[-1]['lr'], clock.epoch)
        scheduler.step(valid_out['epoch_loss'])

        if valid_out['epoch_loss'] < sess.best_val_loss:
            sess.best_val_loss = valid_out['epoch_loss']
            sess.save_checkpoint('best_model.pth.tar')

        if clock.epoch % 10 == 0:
            sess.save_checkpoint('epoch{}.pth.tar'.format(clock.epoch))
        sess.save_checkpoint('latest.pth.tar')

        clock.tock()


if __name__ == '__main__':
    main()
