from alive_progress import alive_it
import torch
import torch.nn as nn
import torch.cuda
import os
import cv2
import numpy as np
import logging
# from apex import amp

from utils.plot import *


class Trainer(object):

    def __init__(self, args, model, criterion, optimizer, trainloader, valloader, device, scheduler=None, fold=None):
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
        self.accumulate_step = args.accumulate_step
        self.best_train_epoch, self.best_val_epoch = 0, 0
        self.best_train_dice, self.best_val_dice = 0., 0.
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

        # train_loss_curve = list()
        # valid_loss_curve = list()
        # train_dice_curve = list()
        # valid_dice_curve = list()


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
            for pv_imgs, art_imgs, pv_mask, art_mask in self.trainloader:
                batch = pv_imgs.size(0)
                step += 1
                pv_imgs = pv_imgs.to(self.device)
                art_imgs = art_imgs.to(self.device)
                pv_mask = pv_mask.to(self.device)
                art_mask = art_mask.to(self.device)

                # forward -- accumulate gradient
                
                self.optimizer.zero_grad()

                predicts, targets, metric_imgs, metric_targets = self.model(pv_imgs, art_imgs,pv_mask, art_mask)
                loss = self.criterion(predicts, targets) / self.accumulate_step
                
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward(retain_graph=True)
                # loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()


                # metric
                dice = self.dice_metric(metric_imgs, metric_targets)
                train_dice += dice.item()
                train_loss += loss.item()
                # train_loss_curve.append(loss.item() / small_batch_size)

                self.logging.info("fold: %d, %d/%d, train_loss:%0.8f, train_dice:%0.8f" % (
                    self.fold, step, batch_num, loss.item(), dice.item() / batch))
                # print("fold: %d, %d/%d, train_loss:%0.8f, train_dice:%0.8f" % (
                #     self.fold, step, batch_num, loss.item(), dice.item() / batch))

            aver_train_dice = train_dice / dt_size
            aver_train_loss = train_loss / batch_num
            # train_loss_curve.append(aver_train_loss)
            # train_dice_curve.append(aver_train_dice)

            if self.best_train_dice < aver_train_dice:
                self.best_train_dice = aver_train_dice
                self.best_train_epoch = epoch
            self.logging.info("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))
            print("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))

            # Validate
            aver_val_loss, aver_val_dice = self.val_epoch()
            # valid_loss_curve.append(aver_val_loss)
            # valid_dice_curve.append(aver_val_dice)
            self.logging.info("epoch %d aver_valid_loss:%0.8f, aver_valid_dice:%0.8f" % (epoch, aver_val_loss, aver_val_dice))
            print("epoch %d aver_valid_loss:%0.8f, aver_valid_dice:%0.8f" % (epoch, aver_val_loss, aver_val_dice))

            # save model weight
            weights_path = os.path.join(self.args.weights_dir, self.args.arch,'batch_size_' + str(self.args.batch_size))
            if not os.path.exists(weights_path):
                os.makedirs(weights_path)
            if self.best_val_dice < aver_val_dice:
                best_val_weight_path = os.path.join(weights_path, 'fold_' + str(self.fold) + '_best_val_dice.pth')
                torch.save(self.model.module.state_dict(), best_val_weight_path)
                self.best_val_dice = aver_val_dice
                self.best_val_epoch = epoch

            # if (epoch + 1) % 2 == 0:
            #     filename = 'fold_' + str(self.fold) + '_epochs_' + str(epoch + 1) + '.pth'
            #     weight_path = os.path.join(weights_path, filename)
            #     torch.save(self.model.module.state_dict(), weight_path)

            self.logging.info("epoch:%d best_train_dice:%0.8f, best_train_epoch:%d, best_valid_dice:%0.8f, best_val_epoch:%d"
                % (epoch, self.best_train_dice, self.best_train_epoch, self.best_val_dice, self.best_val_epoch))

            # scheduler
            if self.scheduler is not None:
                self.scheduler.step(aver_val_dice)
                # self.scheduler.step()

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
            loss_v, dice_v, ii = 0., 0., 0
            dt_size = len(self.valloader.dataset)
            batch_num = (dt_size - 1) // self.valloader.batch_size + 1

            # validation
            for pv_imgs, art_imgs, pv_mask, art_mask in self.valloader:

                pv_imgs = pv_imgs.to(self.device)
                art_imgs = art_imgs.to(self.device)
                pv_mask = pv_mask.to(self.device)
                art_mask = art_mask.to(self.device)

                predicts, targets, metric_imgs, metric_targets = self.model(pv_imgs, art_imgs, pv_mask, art_mask)
                loss = self.criterion(predicts, targets)
                
                loss_v += loss.item()
                dice_v += self.dice_metric(metric_imgs, metric_targets)

                # save prediction
                # metric_imgs = metric_imgs > 0.5
                # metric_targets = metric_targets > 0
                # for num in range(pv_imgs.shape[0]):
                #     index = pv_imgs.shape[1] // 2
                #     x = torch.squeeze(pv_imgs[num, index, :, :]).cpu().numpy()
                #     output = torch.squeeze(metric_imgs[num, 0, :, :]).cpu().numpy()
                #     ground = torch.squeeze(metric_targets[num, 0, :, :]).cpu().numpy()
                #     src_path = os.path.join(save_root, "predict_%d_origin.png" % ii)
                #     output_path = os.path.join(save_root, "predict_%d_predict.png" % ii)
                #     ground_path = os.path.join(save_root, "predict_%d_mask.png" % ii)

                #     cv2.imwrite(src_path, x * 255)
                #     cv2.imwrite(output_path, output * 255)
                #     cv2.imwrite(ground_path, ground * 255)
                #     ii += 1

        return loss_v / batch_num, dice_v / dt_size


    def dice_metric(self, predicts, targets):

        smooth = 1e-5

        # predicts = torch.round(torch.sigmoid(predicts))
        predicts = (predicts > 0.5).float()
        targets = (targets > 0).float()

        N = targets.size(0)
        pred_flat = predicts.view(N, -1)
        gt_flat = targets.view(N, -1)
    
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        dice = (2 * intersection + smooth) / (unionset + smooth)

        return dice.sum()
    
    