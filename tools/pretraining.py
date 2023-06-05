# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-03-03 22:30:21
#
# Created on Mon Oct 18 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import os
import argparse
from statistics import mode
import yaml
import sys

#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


sys.path.append('../')
sys.path.append('../../')
sys.path.append('./')
from tools.utils import logging_sit, save_reconstruction_mae, get_data_path, get_dataloaders, get_dimensions, get_scheduler

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.sit import SiT
from models.mpp import masked_patch_pretraining
from models.mae import MAE

from torch.utils.tensorboard import SummaryWriter


def train(config):


    #############################
    ######     CONFIG      ######
    #############################

    print('')
    print('#'*30)
    print('########### Config ###########')
    print('#'*30)
    print('')

    #mesh_resolution
    ico_mesh = config['mesh_resolution']['ico_mesh']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']
    sampling = config['mesh_resolution']['sampling']

    #data
    dataset = config['data']['dataset']
    task = config['data']['task']
    configuration = config['data']['configuration']
    hemi = config['data']['hemi']

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    epochs = config['training']['epochs']
    val_epochs = config['training']['val_epoch']
    log_training_epoch = config['training']['log_training_epoch']
    log_iteration = config['training']['log_iteration']
    use_confounds = config['training']['use_confounds']
    early_stopping = config['training']['early_stopping']

    if config['MODEL'] == 'sit':    
        channels = config['transformer']['channels']
    num_channels = len(channels)

    if hemi == 'full':
        num_patches*=2

    #data path
    try:
        data_path = get_data_path(config)
    except:
        raise("can't get data path")

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('method: {}'.format(config['SSL']))
    print('gpu: {}'.format(device))   
    print('dataset: {}'.format(dataset))  
    print('task: {}'.format(task))  
    print('model: {}'.format(config['MODEL']))
    print('configuration: {}'.format(configuration))  
    print('data path: {}'.format(data_path))

    ##############################
    ######     DATASET      ######
    ##############################

    print('')
    print('#'*30)
    print('######## Loading data ########')
    print('#'*30)
    print('')
    if config['MODEL'] == 'sit':
        print('Mesh resolution - ico {}'.format(ico_mesh))
        print('Grid resolution - ico {}'.format(ico_grid))
        print('Number of patches - {}'.format(num_patches))
        print('Number of vertices - {}'.format(num_vertices))
        print('')

    try:
        train_loader, val_loader, test_loader = get_dataloaders(config,data_path)
    except:
        raise("can't get dataloaders")


    ##############################
    ######      LOGGING     ######
    ##############################

    print('')
    print('#'*30)
    print('########## Logging ###########')
    print('#'*30)
    print('')

    # creating folders for logging. 
    if config['MODEL'] == 'sit':
        folder_to_save_model = logging_sit(config,pretraining=True)
    else:
        raise('not implemented yet')
    
    try:
        os.makedirs(folder_to_save_model,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))
    
    #tensorboard
    writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################

    print('')
    print('#'*30)
    print('######### Init model #########')
    print('#'*30)
    print('')

    T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config)

    if config['MODEL'] == 'sit':

        model = SiT(dim=config['transformer']['dim'],
                        depth=config['transformer']['depth'],
                        heads=config['transformer']['heads'],
                        pool=config['transformer']['pool'], 
                        num_patches=N,
                        num_classes=config['transformer']['num_classes'],
                        num_channels=T,
                        num_vertices=V,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'],
                        use_pe=config['transformer']['use_pos_embedding'],
                        bottleneck_dropout=bottleneck_dropout,
                        use_bottleneck=use_bottleneck,
                        use_confounds=use_confounds,
                        weights_init=config['transformer']['init_weights'],
                        use_class_token=config['transformer']['use_class_token'],
                        trainable_pos_emb=config['transformer']['trainable_pos_emb'])


    if config['training']['restart']:
        checkpoint = torch.load(config['training']['path_from_ckpt'])
        model.load_state_dict(checkpoint['model_state_dict']) 
        epoch_to_start = checkpoint['epoch']
        print('Starting training from epoch  {}'.format(epoch_to_start))
    else:
        epoch_to_start= 0
        print('Training from scratch')

    model.to(device)

    print('Number of parameters encoder: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    ##################################################
    #######     SELF-SUPERVISION PIPELINE      #######
    ##################################################

    if config['SSL'] == 'mpp':

        print('Pretrain using Masked Patch Prediction')
        ssl = masked_patch_pretraining(transformer=model,
                                    dim_in = config['transformer']['dim'],
                                    dim_out= V*T,
                                    device=device,
                                    mask_prob=config['pretraining_mpp']['mask_prob'],
                                    replace_prob=config['pretraining_mpp']['replace_prob'],
                                    swap_prob=config['pretraining_mpp']['swap_prob'],
                                    num_vertices=num_vertices,
                                    channels=T)
    elif config['SSL'] == 'mae':

        print('Pretrain using Masked AutoEncoder')
        ssl = MAE(encoder=model, 
                    masking_ratio=config['pretraining_mae']['mask_prob'],
                    decoder_dim=config['pretraining_mae']['decoder_dim'],
                    decoder_depth=config['pretraining_mae']['decoder_depth'],
                    decoder_heads = config['pretraining_mae']['decoder_heads'],
                    decoder_dim_head = config['pretraining_mae']['decoder_dim_head'],
                    use_pos_embedding_decoder=config['pretraining_mae']['use_pos_embedding_decoder'],
                    use_all_patch_loss= config['pretraining_mae']['use_all_patch_loss'],
                    loss=config['pretraining_mae']['loss'],
                    mask=config['data']['masking'],
                    dataset=dataset,
                    configuration = configuration,
                    sampling=sampling,
                    sub_ico=ico_grid,
                    num_channels=num_channels,
                    weights_init=config['pretraining_mae']['init_weights'],
                    path_to_template=config['data']['path_to_template'],
                    path_to_workdir = config['data']['path_to_workdir'])
    else:
        raise('not implemented yet')  
    
    ssl.to(device)

    print('Number of parameters pretraining pipeline : {:,}'.format(sum(p.numel() for p in ssl.parameters() if p.requires_grad)))
    print('')

    #####################################
    #######     OPTIMISATION      #######
    #####################################

    if config['optimisation']['optimiser']=='Adam':
        print('using Adam optimiser')
        optimizer = optim.Adam(ssl.parameters(), lr=LR, 
                                                weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        print('using SGD optimiser')
        optimizer = optim.SGD(ssl.parameters(), lr=LR, 
                                                momentum=config['SGD']['momentum'],
                                                weight_decay=config['SGD']['weight_decay'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        print('using AdamW optimiser')
        optimizer = optim.AdamW(ssl.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')

    if config['training']['restart']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    ###################################
    #######     SCHEDULING      #######
    ###################################
    
    it_per_epoch = np.ceil(len(train_loader))

    scheduler = get_scheduler(config, it_per_epoch, optimizer)

    ##################################
    ######     PRE-TRAINING     ######
    ##################################

    print('')
    print('#'*30)
    print('#### Starting pre-training ###')
    print('#'*30)
    print('')

    best_val_loss = 100000000000
    c_early_stop = 0
    
    for epoch in range(epochs):

        ssl.train()

        running_loss = 0

        for i, data in enumerate(train_loader):

            inputs, _ = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            if config['SSL'] == 'mae':
                mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices= ssl(inputs)
            elif config['SSL'] == 'mpp':
                mpp_loss, _ = ssl(inputs)
                
            mpp_loss.backward()
            optimizer.step()

            running_loss += mpp_loss.item()

            writer.add_scalar('loss/train_it', mpp_loss.item(), epoch*it_per_epoch + i + 1 + epoch_to_start)

            if config['optimisation']['use_scheduler']:
                scheduler.step()
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i + 1 + epoch_to_start )
            else:
                if config['optimisation']['warmup']:
                    scheduler.step()
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 + epoch_to_start)
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 + epoch_to_start)

            ##############################
            #########  LOG IT  ###########
            ##############################

            if (epoch*it_per_epoch + i+1)%log_iteration==0:

                loss_pretrain_it = running_loss / (i+1)

                if config['optimisation']['use_scheduler']:
                    print('| It - {} | Loss - {:.4f} | LR - {}'.format(epoch*it_per_epoch + i +1+ epoch_to_start, loss_pretrain_it, scheduler.get_last_lr()[0] ))
                else:
                    print('| It - {} | Loss - {:.4f} | LR - {}'.format(epoch*it_per_epoch + i +1+ epoch_to_start, loss_pretrain_it, optimizer.param_groups[0]['lr']))
            
                if config['SSL'] == 'mae' and config['pretraining_mae']['save_reconstruction']:
                    #print('saving reconstruction')
                    save_reconstruction_mae(reconstructed_batch.detach()[:1],
                                            reconstructed_batch_unmasked.detach()[:1],
                                                inputs, 
                                                num_patches,
                                                num_vertices,
                                                ico_grid,
                                                num_channels,
                                                masked_indices[:1],
                                                unmasked_indices[:1],
                                                str(int(epoch*it_per_epoch + i +1+ epoch_to_start)).zfill(6),
                                                folder_to_save_model,
                                                split='train',
                                                path_to_workdir=config['data']['path_to_workdir'],
                                                id='0',
                                                server = config['SERVER']
                                                )

        
        loss_pretrain_epoch = running_loss / (i+1)

        writer.add_scalar('loss/train', loss_pretrain_epoch, epoch+1+epoch_to_start)

        if (epoch+1)%log_training_epoch==0:

            if config['optimisation']['use_scheduler']:
                print('| Epoch - {} | Loss - {} | LR - {}'.format(epoch+1+epoch_to_start, loss_pretrain_epoch, scheduler.get_last_lr()[0] ))
            else:
                print('| Epoch - {} | Loss - {} | LR - {}'.format(epoch+1+epoch_to_start, loss_pretrain_epoch, optimizer.param_groups[0]['lr']))

        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epochs==0: 
            
            running_val_loss = 0
            ssl.eval()

            with torch.no_grad():

                for i, data in enumerate(val_loader):

                    inputs, _ = data[0].to(device), data[1].to(device)

                    if config['SSL'] == 'mae':
                        mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices = ssl(inputs)
                    elif config['SSL'] == 'mpp':
                        mpp_loss, _ = ssl(inputs)

                    running_val_loss += mpp_loss.item()

            loss_pretrain_val_epoch = running_val_loss /(i+1)

            writer.add_scalar('loss/val', loss_pretrain_val_epoch, epoch+1+epoch_to_start)

            print('| Validation | Epoch - {} | Loss - {} | '.format(epoch+1+epoch_to_start, loss_pretrain_val_epoch))

            if loss_pretrain_val_epoch < best_val_loss:
                best_val_loss = loss_pretrain_val_epoch
                best_epoch = epoch+1+epoch_to_start
                c_early_stop = 0

                if config['SSL'] == 'mae' and config['pretraining_mae']['save_reconstruction']:
                    for i, data in enumerate(val_loader):
                        if i<3:

                            inputs, _ = data[0].to(device), data[1].to(device)

                            mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices = ssl(inputs)

                            save_reconstruction_mae(
                                                    reconstructed_batch.detach(),
                                                    reconstructed_batch_unmasked.detach(),    
                                                    inputs, 
                                                    num_patches,
                                                    num_vertices,
                                                    ico_grid,
                                                    num_channels,
                                                    masked_indices,
                                                    unmasked_indices,
                                                    str(int(epoch+1)).zfill(6),
                                                    folder_to_save_model,
                                                    split='val',
                                                    path_to_workdir=config['data']['path_to_workdir'],
                                                    id = i,
                                                    server = config['SERVER']
                                                    )

                config['results'] = {}
                config['results']['best_epoch'] = best_epoch
                config['results']['best_current_loss'] = loss_pretrain_epoch
                config['results']['best_current_loss_validation'] = best_val_loss

                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                        yaml.dump(config, yaml_file)

                #print('saving_model')
                torch.save({ 'epoch':epoch+1+epoch_to_start,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss':loss_pretrain_epoch,
                             },
                            os.path.join(folder_to_save_model, 'encoder-best.pt'))
                torch.save({ 'epoch':epoch+1+epoch_to_start,
                             'model_state_dict': ssl.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss':loss_pretrain_epoch,
                             },
                            os.path.join(folder_to_save_model, 'encoder-decoder-best.pt'))
            elif early_stopping:
                c_early_stop += 1

        if early_stopping and (c_early_stop>=early_stopping):
            print('stop training - early stopping')
            break


    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results']['final_loss'] = loss_pretrain_epoch

    if early_stopping and (c_early_stop>=early_stopping):
        config['results']['training_finished'] = 'early stopping' 
    else:
        config['results']['training_finished'] = True 

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    
    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################

    torch.save({'epoch':epoch+1+epoch_to_start,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_epoch,
                },
                os.path.join(folder_to_save_model,'encoder-final.pt'))

    torch.save({'epoch':epoch+1+epoch_to_start,
                'model_state_dict': ssl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_epoch,
                },
                os.path.join(folder_to_save_model,'encoder-decoder-final.pt'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    train(config)


