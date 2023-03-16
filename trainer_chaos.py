from alive_progress import alive_it
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import os
import cv2
import numpy as np
import logging
# from apex import amp

from utils.plot import *


class Trainer(object):

    def __init__(self, args, model, criterion, optimizer, trainloader, valloader, device, scheduler=None, fold=None, expand_size=0):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.scheduler = scheduler
        self.fold = fold
        self.expand_size = expand_size
        self.accumulate_step = args.accumulate_step
        self.logging = Trainer.getLog(self.args)
        self.logging.info("====================\nArgs:{}\n====================".format(self.args))

    
    def getLog(args):
        dirname = os.path.join(args.logs_dir, args.arch, 'epochs_'+str(args.epochs),
            'batch_size_'+str(args.batch_size), 'pos_weight_'+str(args.pos_weight))
        filename = os.path.join(dirname, 'log.log')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        logging.basicConfig(
                filename=filename,
                level=logging.INFO,
                format='%(asctime)s:%(message)s'
            )
        return logging

    def train(self):

        best_train_epoch, best_val_epoch = 0, 0
        best_train_dice, best_val_dice = 0., 0.
        train_loss_curve = list()
        valid_loss_curve = list()
        train_dice_curve = [0., 0., 0., 0., 0.]
        valid_dice_curve = [0., 0., 0., 0., 0.]


        for epoch in alive_it(range(1, self.args.epochs + 1)):

            self.logging.info('-' * 20)
            self.logging.info('Epoch {}/{} lr: {}'.format(epoch, self.args.epochs, self.optimizer.param_groups[0]['lr']))
            self.logging.info('-' * 20)
            dt_size = len(self.trainloader.dataset)
            train_loss = 0
            train_dice = 0
            step = 0
            N = self.trainloader.batch_size
            small_batch_size = N // self.accumulate_step
            batch_num = (dt_size - 1) // N + 1

            # train
            self.model.train()
            for pv_imgs, art_imgs, pv_mask, art_mask, spacing in self.trainloader:
                batch = pv_imgs.size(0)
                step += 1
                pv_imgs = pv_imgs.to(self.device)
                art_imgs = art_imgs.to(self.device)
                pv_mask = pv_mask.to(self.device)
                art_mask = art_mask.to(self.device)


                self.optimizer.zero_grad()

                predicts, targets, metric_imgs, metric_targets = self.model(pv_imgs, art_imgs,pv_mask, art_mask)
                loss = self.criterion(predicts, targets) / self.accumulate_step
                
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward(retain_graph=True)
                # loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()


                # metric
                dice_list = self.multi_dice_metric(metric_imgs, metric_targets)
                dice_all = (dice_list[1] + dice_list[2] +dice_list[3] +dice_list[4]) / 4
                train_dice += dice_all
                train_loss += loss.item()
                train_loss_curve.append(loss.item() / small_batch_size)

                self.logging.info("fold: %d, %d/%d, train_loss:%0.8f, dice_all:%0.8f, train_dice:%0.8f, %0.8f, %0.8f, %0.8f" % (
                    self.fold, step, batch_num, loss.item(), dice_all / batch, dice_list[1] / batch, dice_list[2] / batch, dice_list[3] / batch, dice_list[4] / batch))
                # print("fold: %d, %d/%d, train_loss:%0.8f, train_dice:%0.8f" % (
                #     self.fold, step, batch_num, loss.item(), dice.item() / batch))
            
            aver_train_dice = train_dice / dt_size
            aver_train_loss = train_loss / batch_num
            # train_loss_curve.append(aver_train_loss)
            train_dice_curve.append(aver_train_dice)

            self.logging.info("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))
            print("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))

            # Validate
            # if epoch % 5 == 0:

            aver_val_loss, aver_val_dice, aver_val_dice_all= self.val_epoch()
            valid_loss_curve.append(aver_val_loss)
            valid_dice_curve.append(aver_val_dice_all)
            self.logging.info("epoch %d aver_valid_loss:%0.8f, aver_val_dice_all:%0.8f, aver_valid_dice:%0.8f, %0.8f, %0.8f, %0.8f" 
            % (epoch, aver_val_loss, aver_val_dice_all, aver_val_dice[1], aver_val_dice[2], aver_val_dice[3], aver_val_dice[4]))
            print("epoch %d aver_valid_loss:%0.8f, aver_val_dice_all:%0.8f,  aver_valid_dice:%0.8f, %0.8f, %0.8f, %0.8f" 
            % (epoch, aver_val_loss, aver_val_dice_all, aver_val_dice[1], aver_val_dice[2], aver_val_dice[3], aver_val_dice[4]))

            # save model weight
            weights_path = os.path.join(self.args.weights_dir, self.args.arch,'batch_size_' + str(self.args.batch_size))
            if not os.path.exists(weights_path):
                os.makedirs(weights_path)

            # if (epoch + 1) % 2 == 0:
            #     filename = 'fold_' + str(self.fold) + '_epochs_' + str(epoch + 1) + '.pth'
            #     weight_path = os.path.join(weights_path, filename)
            #     torch.save(self.model.module.state_dict(), weight_path)

            if best_train_dice < aver_train_dice:
                best_train_dice = aver_train_dice
                best_train_epoch = epoch

            if best_val_dice < aver_val_dice_all:
                best_val_weight_path = os.path.join(weights_path, 'fold_' + str(self.fold) + '_best_val_dice.pth')
                torch.save(self.model.module.state_dict(), best_val_weight_path)
                best_val_dice = aver_val_dice_all
                best_val_epoch = epoch

            self.logging.info("epoch:%d best_train_dice:%0.8f, best_train_epoch:%d, best_valid_dice:%0.8f, best_val_epoch:%d"
                % (epoch, best_train_dice, best_train_epoch, best_val_dice, best_val_epoch))

            # scheduler
            if self.scheduler is not None:
                # self.scheduler.step(aver_val_dice_all)
                self.scheduler.step()

        # train_x = range(len(train_loss_curve))
        # train_y = train_loss_curve

        # train_iters = len(self.trainloader)
        # valid_x = np.arange(1, len(valid_loss_curve) + 1) * train_iters
        # valid_y = valid_loss_curve
        # loss_plot(self.args, self.fold, train_x, train_y, valid_x, valid_y)
        # metrics_plot(self.args, self.fold, 'train&valid', train_dice_curve, valid_dice_curve)


    def val_epoch(self):
        save_root = self.args.predicts_dir
        self.model.eval()
        with torch.no_grad():
            loss_v, dice_all, ii = 0., 0., 0
            dice_v = np.array([0., 0., 0., 0., 0.])
            dt_size = len(self.valloader.dataset)
            batch_num = (dt_size - 1) // self.valloader.batch_size + 1

            # validation
            for pv_imgs, art_imgs, pv_masks, art_masks, spacing in self.valloader:

                pv_imgs = pv_imgs.to(self.device)
                art_imgs = art_imgs.to(self.device)
                pv_masks = pv_masks.to(self.device)
                art_masks = art_masks.to(self.device)

                predicts = None
                targets = None
                for i in range(self.expand_size, pv_imgs.size()[2] - self.expand_size):
                    if self.expand_size > 0:
                        pv_img = pv_imgs[:, :, i - self.expand_size: i + self.expand_size + 1, :, :].squeeze(1)
                        art_img = art_imgs[:, :, i - self.expand_size: i + self.expand_size + 1, :, :].squeeze(1)
                    else: 
                        pv_img = pv_imgs[:, :, i, :, :]
                        art_img = art_imgs[:, :, i, :, :]
                    pv_mask = pv_masks[:, :, i, :, :]
                    art_mask = art_masks[:, :, i, :, :]
                    a, b, metric_imgs, metric_targets = self.model(pv_img, art_img, pv_mask, art_mask)
                    loss = self.criterion(a, b)

                    if predicts is None:
                        predicts = metric_imgs.unsqueeze(2)
                        targets = pv_mask.unsqueeze(2)
                    else:
                        predicts = torch.cat([predicts, metric_imgs.unsqueeze(2)], dim=2)
                        targets = torch.cat([targets, pv_mask.unsqueeze(2)], dim=2)

                # ii, loss = 0, 0.
                # B, _, Z, H, W = pv_imgs.size()
                # pv_pred_temp = torch.zeros((B, 5, Z, H, W))
                # for z in range(8, pv_imgs.size(2), 8):
                #     for y in range(64, pv_imgs.size(3), 64):
                #         for x in range(64, pv_imgs.size(4), 64):
                #             if z + 8 > pv_imgs.size(2):
                #                 z = pv_imgs.size(2) - 8
                #             pv_patch = pv_imgs[:, :, z-8:z+8, y-64:y+64, x-64:x+64]
                #             art_patch = art_imgs[:, :, z-8:z+8, y-64:y+64, x-64:x+64]
                #             pv_mask_patch = pv_masks[:, :, z-8:z+8, y-64:y+64, x-64:x+64]
                #             art_mask_patch = art_masks[:, :, z-8:z+8, y-64:y+64, x-64:x+64]
                #             predicts, targets, metric_imgs, metric_targets = self.model(pv_patch, art_patch, pv_mask_patch, art_mask_patch)
                #             loss += self.criterion(predicts, targets)
                #             pv_pred_temp[:, :, z-8:z+8, y-64:y+64, x-64:x+64] += metric_imgs.cpu()
                #             ii += 1
                # loss = loss / ii
                # predicts = pv_pred_temp
                # targets = pv_masks

                loss_v += loss.item()
                dice_list = self.multi_dice_metric(predicts, targets)
                dice_v += dice_list
                dice_all += ((dice_list[1] + dice_list[2] +dice_list[3] +dice_list[4]) / 4)
                
        return loss_v / batch_num, np.array(dice_v) / dt_size, dice_all / dt_size


        
    def multi_dice_metric(self, predicts, targets):

        smooth = 1e-5

        N, C = targets.size(0), targets.size(1)
        targets = torch.cat([targets == 0, targets == 1, targets == 2, targets == 3, targets == 4], dim=1).cpu()
        gt_flat = targets.view(N, 5, -1)
        # print(0, gt_flat.size())

        N, C = predicts.size(0), predicts.size(1)
        predicts = predicts.view(N, C, -1)
        max = torch.argmax(predicts, dim=1, keepdim=True)
        pred_flat = torch.cat([max == 0, max == 1, max == 2, max == 3, max == 4], dim=1).cpu()
        # print(1, pred_flat.size())


        intersection = (pred_flat * gt_flat).sum(2)
        unionset = pred_flat.sum(2) + gt_flat.sum(2)
        dice = (2 * intersection + smooth) / (unionset + smooth)

        # print(dice.size())
        return dice.sum(0).cpu().numpy()

if __name__ == '__main__':

    a = torch.randn((2, 5, 256, 256))
    b = (torch.randn((2, 1, 256, 256)) * 5).int()
    a[a < 0] = 0
    b[b < 0] = 0
    b[b > 4] = 0
    # print(b.size())
    dice = multi_dice_metric(a, b)
    print(dice)