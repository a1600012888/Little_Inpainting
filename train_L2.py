import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
from model import network
from common import config
from utils import TrainClock, save_args, AverageMeter
from dataset import get_dataloaders

torch.backends.cudnn.benchmark = True

Hole_Loss_weight = 6

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


def train_model(train_loader, model, criterion, optimizer, epoch, tb_writer):
    losses = AverageMeter()
    hole_losses = AverageMeter()
    valid_losses = AverageMeter()
    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs = data['hole_img'].float()
        labels = data['ori_img'].float()
        # mask: 1 for the hole and 0 for others
        masks = data['mask'].float()
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        masks = masks.to(config.device)

        # pass this batch through our model and get y_pred
        outputs = model(inputs)

        # update loss metric
        hole_loss = criterion(outputs*masks, labels*masks)
        valid_loss = criterion(outputs*(1-masks), labels*(1-masks))
        loss = hole_loss * Hole_Loss_weight + valid_loss
        losses.update(loss.item(), inputs.size(0))
        hole_losses.update(hole_loss.item(), inputs.size(0))
        valid_losses.update(valid_loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            loss="{:.4f}".format(losses.avg))

    torch.cuda.empty_cache()
    tb_writer.add_scalar('train/epoch_loss', losses.avg, epoch)
    tb_writer.add_scalar('train/hole_loss', hole_losses.avg, epoch)
    tb_writer.add_scalar('train/valid_loss', valid_losses.avg, epoch)

    return


def valid_model(valid_loader, model, criterion, epoch, tb_writer):
    losses = AverageMeter()
    hole_losses = AverageMeter()
    valid_losses = AverageMeter()
    # ensure model is in train mode
    model.eval()
    pbar = tqdm(valid_loader)
    for i, data in enumerate(pbar):
        inputs = data['hole_img'].float()
        labels = data['ori_img'].float()
        # mask: 1 for the hole and 0 for others
        masks = data['mask'].float()
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        masks = masks.to(config.device)

        with torch.no_grad():
            # pass this batch through our model and get y_pred
            outputs = model(inputs)

            # update loss metric
            # suppose criterion is L2 loss
            hole_loss = criterion(outputs*masks, labels*masks)
            valid_loss = criterion(outputs*(1-masks), labels*(1-masks))
            loss = Hole_Loss_weight * hole_loss + valid_loss
            losses.update(loss.item(), inputs.size(0))
            hole_losses.update(hole_loss.item(), inputs.size(0))
            valid_losses.update(valid_loss.item(), inputs.size(0))
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
            loss="{:.4f}".format(losses.avg))

    tb_writer.add_scalar('valid/epoch_loss', losses.avg, epoch)
    tb_writer.add_scalar('valid/hole_loss', hole_losses.avg, epoch)
    tb_writer.add_scalar('valid/valid_loss', valid_losses.avg, epoch)

    torch.cuda.empty_cache()

    outspects = {
        'epoch_loss': losses.avg,
    }
    return outspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='epoch number')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('--exp_name', default=config.exp_name, type=str, required=False)
    args = parser.parse_args()
    print(args)

    config.exp_name = args.exp_name
    config.make_dir()

    save_args(args, config.log_dir)
    net = network()

    net = torch.nn.DataParallel(net).cuda()
    sess = Session(config, net=net)


    train_loader = get_dataloaders(os.path.join(config.data_dir, 'train.json'),
                                   batch_size=args.batch_size, shuffle=True)
    valid_loader = get_dataloaders(os.path.join(config.data_dir, 'val.json'),
                                   batch_size=args.batch_size, shuffle=True)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    clock = sess.clock
    tb_writer = sess.tb_writer

    criterion = nn.MSELoss().cuda()

    optimizer = optim.Adam(sess.net.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    for e in range(args.epochs):
        train_model(train_loader, sess.net,
                                criterion, optimizer, clock.epoch, tb_writer)
        valid_out = valid_model(valid_loader, sess.net,
                                criterion, clock.epoch, tb_writer)
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