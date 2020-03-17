# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers, close_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optimizer
import cv2
import torch.nn.functional as F
from torch.nn import init

# global variables
parser = argument_parser()
args = parser.parse_args()


######################
#  Decoder Networks  #
######################

#####  ResNet  #####
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Use kernel size 4 to make sure deconv(conv(x)) has the same shape as x
# not working well...
# https://distill.pub/2016/deconv-checkerboard/
def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
                  kernel_size=3, stride=1, padding=0)
    )

# Basic resnet block:
# x ---------------- shortcut ---------------x
# \___conv___norm____relu____conv____norm____/
class BasicResBlock(nn.Module):
    def __init__(self, inplanes, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.LeakyReLU(0.2, True)):
        super(BasicResBlock, self).__init__()

        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.inplanes = inplanes

        layers = [
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes),
            activation_layer,
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes)
        ]
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return self.res(x) + x

# ResBlock: A classic ResBlock with 2 conv layers and a up/downsample conv layer. (2+1)
# x ---- BasicConvBlock ---- ReLU ---- conv/upconv ----
# If direction is "down", we use nn.Conv2d with stride > 1, getting a smaller image
# If direction is "up", we use nn.ConvTranspose2d with stride > 1, getting a larger image
class ConvResBlock(nn.Module):
    def __init__(self, inplanes, planes, direction, stride=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(ConvResBlock, self).__init__()
        self.res = BasicResBlock(inplanes, norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        self.activation = activation_layer

        if stride == 1 and inplanes == planes:
            conv = lambda x: x
        else:
            if direction == 'down':
                conv = conv3x3(inplanes, planes, stride=stride)
            elif direction == 'up':
                conv = deconv3x3(inplanes, planes, stride=stride)
            else:
                raise (ValueError('Direction must be either "down" or "up", get %s instead.' % direction))
        self.conv = conv
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.conv(self.activation(self.res(x)))

#####  Decoder #####
class ConvResDecoder(nn.Module):
    '''
        ConvResDecoder: Use convres block for upsampling
    '''

    def __init__(self):
        super(ConvResDecoder, self).__init__()

        # Xin Jin: this is R-50 inter-channel (2048) with last_stride = 1
        input_channel = 2048
        final_channel = 16 # 16

        # For UNet structure:
        self.embed_layer3 = nn.Sequential(
                        nn.Conv2d(in_channels=1024, out_channels=512,
                                  kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True)
                    )
        self.embed_layer2 = nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=256,
                                  kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
        self.embed_layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=64,
                                  kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                    )

        self.reduce_dim = nn.Sequential(
                          nn.Conv2d(input_channel, input_channel//4, kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(512),
                          nn.ReLU(inplace=True)
                    )     # torch.Size([64, 512, 16, 8])

        self.up1 = ConvResBlock(512, 256, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True)) # torch.Size([64, 256, 32, 16])
        self.up2 = ConvResBlock(256, 64, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))  # torch.Size([64, 64, 64, 32])
        self.up3 = ConvResBlock(64, 32, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))   # torch.Size([64, 32, 128, 64])
        self.up4 = ConvResBlock(32, 16, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))   # torch.Size([64, 16, 256, 128])

        self.final_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(final_channel, 3, kernel_size=7, stride=1, padding=0)  # torch.Size([64, 3, 256, 128])
            #nn.Tanh()
            )

    def forward(self, x, x_down1, x_down2, x_down3):

        x_reduce_dim = self.reduce_dim(x)          # torch.Size([64, 512, 16, 8])
        embed_layer3 = self.embed_layer3(x_down3)  # torch.Size([64, 512, 16, 8])
        x = self.up1(embed_layer3 + x_reduce_dim)  # torch.Size([64, 256, 32, 16])
        x_sim1 = x
        embed_layer2 = self.embed_layer2(x_down2)  # torch.Size([64, 256, 32, 16])
        x = self.up2(embed_layer2 + x)             # torch.Size([64, 64, 64, 32])
        x_sim2 = x
        embed_layer1 = self.embed_layer1(x_down1)  # torch.Size([64, 64, 64, 32])
        x = self.up3(embed_layer1 + x)             # torch.Size([64, 32, 128, 64])
        x_sim3 = x
        x = self.up4(x)                            # torch.Size([64, 16, 256, 128])
        x_sim4 = x
        x = self.final_layer(x)                    # torch.Size([64, 3, 256, 128])

        # reconstruct the original size, by Jinx:
        x = F.interpolate(x, size=(x.size(2), x.size(3)*2), mode='bilinear', align_corners=True)
        return x, x_sim1, x_sim2, x_sim3, x_sim4



def init_decoder_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# Main #:
def main():
    global args
    
    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.3f} M".format(count_num_param(model)))

    # Add by Xin Jin, SAN-decoder:
    print("Initializing texture decoder model")
    model_decoder = ConvResDecoder()
    init_decoder_weights(model_decoder, init_type='xavier', gain=0.02)
    print("Decoder Model size: {:.3f} M".format(count_num_param(model_decoder)))

    # Loss:
    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)
    
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    # warm_up settings:
    optimizer_warmup = torch.optim.Adam(model.parameters(), lr=8e-06, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler_warmup = lr_scheduler.ExponentialLR(optimizer_warmup, gamma=1.259)

    # optimize for texture prediction:
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999))


    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        model_decoder = nn.DataParallel(model_decoder).cuda()

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, model_decoder, queryloader, galleryloader, use_gpu, 0, return_distmat=True)
        
            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("==> Start training")

    # Xin Jin: for warming up the model
    if args.warm_up_epoch != 0:
        print("Train the whole model for {} epochs with small lr".format(args.warm_up_epoch))
        for epoch in range(args.warm_up_epoch):
            start_train_time = time.time()
            train(epoch, model, model_decoder, criterion_xent, criterion_htri, optimizer_warmup, optimizer_decoder, optimizer_encoder, trainloader, use_gpu)
            train_time += round(time.time() - start_train_time)
            scheduler_warmup.step()
        print("Warm up Done. All layers are warmed for {} epochs".format(args.warm_up_epoch))


    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, model_decoder, criterion_xent, criterion_htri, optimizer, optimizer_decoder, optimizer_encoder, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()
        
        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:

            print("==> Test")
            
            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, model_decoder, queryloader, galleryloader, use_gpu, epoch)
                ranklogger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict = model.module.state_dict()
                decoder_state_dict = model_decoder.module.state_dict()
            else:
                state_dict = model.state_dict()
                decoder_state_dict = model_decoder.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'decoder_state_dict': decoder_state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()


def train(epoch, model, model_decoder, criterion_xent, criterion_htri, optimizer, optimizer_decoder, optimizer_encoder, trainloader, use_gpu, fixbase=False):
    losses = AverageMeter()
    losses_recon = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    model_decoder.train()

    if fixbase or args.fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, img_paths, imgs_texture) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids, imgs_texture = imgs.cuda(), pids.cuda(), imgs_texture.cuda()
        
        outputs, features, feat_texture, x_down1, x_down2, x_down3 = model(imgs)
        torch.cuda.empty_cache()

        if args.htri_only:
            if isinstance(features, (tuple, list)):
                loss = DeepSupervision(criterion_htri, features, pids)
            else:
                loss = criterion_htri(features, pids)
        else:
            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)
            
            if isinstance(features, (tuple, list)):
                htri_loss = DeepSupervision(criterion_htri, features, pids)
            else:
                htri_loss = criterion_htri(features, pids)
            
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        del outputs, features

        # Second forward for training texture reconstruction
        close_specified_layers(model, ['fc','classifier'])

        recon_texture, x_sim1, x_sim2, x_sim3, x_sim4 = model_decoder(feat_texture, x_down1, x_down2, x_down3)
        torch.cuda.empty_cache()

        loss_rec = nn.L1Loss()
        loss_tri = nn.MSELoss()
        loss_recon = loss_rec(recon_texture, imgs_texture)#*0.1

        # L1 loss to push same id's feat more similar:
        loss_triplet_id_sim1 = 0.0
        loss_triplet_id_sim2 = 0.0
        loss_triplet_id_sim3 = 0.0
        loss_triplet_id_sim4 = 0.0
  
        for i in range(0, ((args.train_batch_size//args.num_instances)-1)*args.num_instances, args.num_instances):
            loss_triplet_id_sim1 += max(loss_tri(x_sim1[i], x_sim1[i+1])-loss_tri(x_sim1[i], x_sim1[i+4])+0.3, 0.0)
            loss_triplet_id_sim2 += max(loss_tri(x_sim2[i+1], x_sim2[i+2])-loss_tri(x_sim2[i+1], x_sim2[i+5])+0.3, 0.0)#loss_tri(x_sim2[i+1], x_sim2[i+2])
            loss_triplet_id_sim3 += max(loss_tri(x_sim3[i+2], x_sim3[i+3])-loss_tri(x_sim3[i+2], x_sim3[i+6])+0.3, 0.0)#loss_tri(x_sim3[i+2], x_sim3[i+3])
            loss_triplet_id_sim4 += max(loss_tri(x_sim4[i], x_sim4[i+3])-loss_tri(x_sim4[i+3], x_sim4[i+4])+0.3, 0.0)#loss_tri(x_sim4[i], x_sim4[i+3])
        loss_same_id = loss_triplet_id_sim1 + loss_triplet_id_sim2 + loss_triplet_id_sim3 + loss_triplet_id_sim4

        loss_recon += (loss_same_id)# * 0.0001)

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss_recon.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        del feat_texture, x_down1, x_down2, x_down3, recon_texture

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))
        losses_recon.update(loss_recon.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_recon {loss_recon.val:.4f} ({loss_recon.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_recon=losses_recon))
        
        end = time.time()
        open_all_layers(model)

        if (epoch + 1) % 50 == 0 :
            print("==> Test reconstruction effect")
            model.eval()
            model_decoder.eval()
            features, feat_texture = model(imgs)
            recon_texture = model_decoder(feat_texture)
            out = recon_texture.data.cpu().numpy()[0].squeeze()
            out = out.transpose((1, 2, 0))
            out = (out / 2.0 + 0.5) * 255.
            out = out.astype(np.uint8)
            print('finish: ', os.path.join(args.save_dir, img_paths[0].split('bounding_box_train/')[-1].split('.jpg')[0]+'ep_'+str(epoch)+'.jpg'))
            cv2.imwrite(os.path.join(args.save_dir, img_paths[0].split('bounding_box_train/')[-1].split('.jpg')[0]+'ep_'+str(epoch)+'.jpg'), out[:,:,::-1])
            model.train()
            model_decoder.train()


def test(model, model_decoder, queryloader, galleryloader, use_gpu, epoch, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()
    model_decoder.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, img_paths, imgs_texture) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, feat_texture, x_down1, x_down2, x_down3 = model(imgs)

            recon_texture, x_sim1, x_sim2, x_sim3, x_sim4 = model_decoder(feat_texture, x_down1, x_down2, x_down3)
            out = recon_texture.data.cpu().numpy()[0].squeeze()
            out = out.transpose((1, 2, 0))
            out = (out / 2.0 + 0.5) * 255.
            out = out.astype(np.uint8)
            print('finish: ', os.path.join(args.save_dir, img_paths[0].split('images_labeled/')[-1].split('.jpg')[0]+'_ep_'+str(epoch)+'.jpg'))
            cv2.imwrite(os.path.join(args.save_dir, img_paths[0].split('images_labeled/')[-1].split('.jpg')[0]+'_ep_'+str(epoch)+'.jpg'), out)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _, imgs_texture) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            
            end = time.time()
            features, feat_texture, x_down1, x_down2, x_down3 = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
