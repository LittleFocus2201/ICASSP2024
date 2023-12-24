import os
import time
import datetime
import csv
import numpy as np
import logging
import sys
import torch
from torch import optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import SecularDataset
import utils
import network as network
from argument import arg as arg
from model.pytorch_SSIM import SSIM
# from apex import amp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def write_losses(file_name, losses_accu, epoch, duration, lrate=None):
    if lrate is None:
        lrate = ['nan']
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 0:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration'] + ['lrate'
                                                                                                                 for lr
                                                                                                                 in
                                                                                                                 lrate]
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg) for loss_avg in losses_accu.values()] + \
                       ['{:.0f}'.format(duration)] + ['{}'.format(lr) for lr in lrate]
        writer.writerow(row_to_write)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def train(opt):
    cudnn.benchmark = opt.cudnn_benchmark
    mixed_precision = opt.mixed_precision
    save_path = opt.training_path
    runs_folder = os.path.join(save_path, opt.runs_name)
    sample_path = os.path.join(runs_folder, opt.sample_path)
    checkpointpath = os.path.join(runs_folder, 'checkpoint')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath)

    # create network
    generator = network.GateGenerator(opt)
    discriminator = network.PatchDiscriminate(opt)
    perceptualnet = network.PerceptualNet()

    # optim
    optimizer_g = optim.AdamW(generator.parameters(), lr=opt.lr_g,weight_decay=0.002)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=opt.lr_d,weight_decay=0.002)

    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.8, patience=4, verbose=True)
    sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.8, patience=4, verbose=True)


    # loss func
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    SSIMloss = SSIM()
    Colorloss = utils.Colorloss()
    Colorloss = Colorloss.cuda()

    # save model

    def save_model(net, epoch, optimizer, opt):
        model_name = '{}_epoch{}_batchsize{}.mdl'.format(net.module._get_name(),epoch, opt.batch_size)
        model_name = os.path.join(checkpointpath, model_name)
        checkpoint = {
            'epoch': epoch,
            'model': net.module.state_dict() if opt.ngpu > 1 else net.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict()
        }
        if epoch % opt.checkpoint_interval == 0:
            torch.save(checkpoint, model_name)
            print('Saving trained model successfully, epoch %d' % epoch)

    def load_model(net, checkpoint):
        state_dict = torch.load(checkpoint)
        net.load_state_dict(state_dict['model'])


    # restart training
    if opt.resume:
        load_model(generator, opt.checkpoint)
        # load_model(discriminator, opt.checkpoint_D)
        startepoch = opt.resume_epoch
    else:
        startepoch = 0

    # to device
    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)
    perceptualnet = torch.nn.DataParallel(perceptualnet)
    generator = generator.cuda()
    discriminator = discriminator.cuda()

    perceptualnet = perceptualnet

    # ----------------------------------
    #  dataload
    # ----------------------------------
    tf_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()])
    trainset = SecularDataset(transform=tf_aug, opt=opt, mode='train')
    print('the number of train images:%d' % len(trainset))
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             pin_memory=True, drop_last=True)
    valset = SecularDataset(transform=tf_aug, opt=opt, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=opt.num_workers,
                           pin_memory=True, drop_last=True)

    # ----------------------------------
    #  Logging
    # ----------------------------------
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(runs_folder, f'experiment.log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    logging.info('Generator model: {}\n'.format(str(generator)))
    logging.info(f'\nStarting Datases\n trainset: {len(trainloader)} \n valset:{len(valloader)} ')

    # ----------------------------------
    #  training
    # ----------------------------------

    # start time
    prev_time = time.time()

    # training
    best_val_loss = 999999
    best_epoch = 0

    for epoch in range(startepoch, opt.epochs):
        logging.info(f'\nbest_epoch: [{best_epoch}]')
        generator.train()
        discriminator.train()

        avg_d_loss = 0.
        avg_content_loss = 0.
        avg_p_loss = 0.
        avg_ssim = 0.
        start_time = time.time()

        for batch_idx, (specular_image, raw_img) in enumerate(tqdm(trainloader)):
            specular_image = specular_image.cuda()
            raw_img = raw_img.cuda()
            
            # print(mask.shape)
            first_out,second_out,HFE = generator(specular_image)
            
            ### train dsicriminator
            optimizer_d.zero_grad()

            # generate out
            first_out = specular_image * (1 - HFE) + first_out * HFE
            second_out = specular_image * (1 - HFE) + second_out * HFE

            # fake samples
            fake_scalar = discriminator(second_out.detach(), HFE)
            # true samples
            true_scalar = discriminator(raw_img, HFE)

            # loss + optim
            hinge_loss = torch.mean(nn.ReLU()(1 - true_scalar)) + torch.mean(nn.ReLU()(fake_scalar + 1))

            loss_D = hinge_loss
            loss_D.backward(retain_graph=True)
            optimizer_d.step()
            
            ### train generator
            optimizer_g.zero_grad()
            # ssimloss
            ssim_ = SSIMloss(first_out, raw_img)
            ssimloss = 1 - ssim_

            # L1loss
            first_Content_loss = (first_out - raw_img).abs().mean()
            second_Content_loss = (second_out - raw_img).abs().mean()

            # Content_loss
            Content_loss = (first_Content_loss + second_Content_loss) #+ MSEloss(first_out, raw_img)
  
            # gan loss
            fake_scalar = discriminator(second_out.detach(), HFE)
            GAN_loss = torch.mean(fake_scalar).abs()

            # perceptual loss
            raw_featuremaps = perceptualnet(raw_img)
            second_out_featuremaps = perceptualnet(second_out)
            perceptualloss = L1loss(raw_featuremaps, second_out_featuremaps)

            # overall loss
            loss = opt.lambda_content * (Content_loss) + opt.lambda_perceptual * perceptualloss + opt.lambda_ssim * ssimloss + opt.lambda_gan * GAN_loss # + \

            loss.backward()
            optimizer_g.step()




            # estimate time
            batches_done = epoch * len(trainloader) + batch_idx
            batches_left = opt.epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # print log
            print("\r[Epoch %d/%d] [Batch %d/%d]  [Content Loss: %.8f] [Gan Loss: %.8f] [loss_D: %.8f]" %
                  ((epoch + 1), opt.epochs, batch_idx, len(trainloader), Content_loss.item(), GAN_loss.item(),loss_D.item()))
            print("\r [Perceptual Loss: %.8f] [SSIM: %.8f] time_left: %s" %
                  (perceptualloss.item(), ssim_.item(), time_left))

            if (batch_idx + 1) % 40 == 0:
                img_list = [raw_img, specular_image, first_out,second_out,HFE]
                name_list = ['train_gt', 'train_specular_image', 'train_first_out','train_second_out','train_HFE']
                utils.save_sample_png(sample_folder=sample_path, sample_name='epoch%d' % (epoch + 1), img_list=img_list,
                                      name_list=name_list, pixel_max_cnt=255)

            # calculate avg losses
            avg_d_loss += float(loss_D)
            avg_content_loss += float(Content_loss)
            avg_p_loss += float(perceptualloss)
            avg_ssim += float(ssim_)



        avg_d_loss /= len(trainloader)
        avg_content_loss /= len(trainloader)
        avg_p_loss /= len(trainloader)
        avg_ssim /= len(trainloader)

        logging.info('Epoch {} Avg_d_loss{} Avg_content_loss{} Avg_p_loss{}'.format(epoch, avg_d_loss, avg_content_loss,
                                                                                    avg_p_loss))
        logging.info('-' * 40)
        training_losses = {
            'avg_content_loss': avg_content_loss,
            'avg_d_loss': avg_d_loss,
            'avg_p_loss': avg_p_loss,
            'avg_ssim': avg_ssim,

        }
        train_duration = time.time() - start_time

        write_losses(os.path.join(runs_folder, 'train.csv'), training_losses, epoch, train_duration)

        generator.eval()
        avg_val_content_loss = 0
        avg_val_ssim = 0
        start_time = time.time()

        for batch_idx, (specular_val_img, raw_val_img) in enumerate(tqdm(valloader)):
            specular_val_img = specular_val_img.cuda()
            raw_val_img = raw_val_img.cuda()



            first_val_out,second_val_out,HFE_val = generator(specular_val_img)

            # generate out
            first_val_out = specular_val_img * (1 - HFE_val) + first_val_out * HFE_val
            second_val_out = specular_val_img * (1 - HFE_val) + second_val_out * HFE_val

            ssim_val = SSIMloss(first_val_out, raw_val_img)

            Content_val_loss = MSEloss(first_val_out, raw_val_img) + L1loss(first_val_out, raw_val_img)
            avg_val_content_loss += float(Content_val_loss)
            avg_val_ssim += float(ssim_val)

        avg_val_content_loss /= len(valloader)
        avg_val_ssim /= len(valloader)
        if avg_val_content_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = avg_val_content_loss
            save_model(generator, (epoch + 1),optimizer_g, opt)
            save_model(discriminator, (epoch + 1),optimizer_d, opt)

        if epoch % 10 ==0:
            save_model(generator, (epoch + 1),optimizer_g, opt)

        val_duration = time.time() - start_time
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, val_duration))
        logging.info('Epoch {} Avg_val_content_loss {} '.format(epoch, avg_val_content_loss))
        val_losses = {
            'Avg_val_content_loss': avg_val_content_loss,
            'ssim_val': avg_val_ssim,

        }

        lr_g = get_lr(optimizer_g)

        lr_d = get_lr(optimizer_d)

        lrate = [lr_g, lr_d]
        write_losses(os.path.join(runs_folder, 'val.csv'), val_losses, epoch, val_duration, lrate)

        # adjust learning rate
        sched_g.step(avg_val_content_loss+avg_val_ssim)
        sched_d.step(avg_val_content_loss+avg_val_ssim)

        ### sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [raw_val_img, specular_val_img, first_val_out,second_val_out,HFE_val]
            name_list = ['val_gt', 'val_specular_img', 'val_first_out','val_second_val','val_HFE']
            utils.save_sample_png(sample_folder=sample_path, sample_name='epoch%d' % (epoch + 1), img_list=img_list,
                                  name_list=name_list, pixel_max_cnt=255)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = arg()
    train(opt)
