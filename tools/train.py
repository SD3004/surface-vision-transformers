# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2022-01-12 15:42:12
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file implements the training procedure to train a SiT model.
Models can be either trained:
    - from scratch
    - from pretrained weights (after self-supervision or ImageNet for instance)

Pretrained ImageNet models are downloaded from the Timm library. 
'''

import os
import argparse
import yaml
import sys
import timm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
sys.path.append('./')

from models.sit import SiT
from tools.utils import load_weights_imagenet




def train(config):

    gpu = config['training']['gpu']
    LR = config['training']['LR']
    use_l1loss = config['training']['l1loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    testing = config['training']['testing']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']

    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    data_path = config['data']['data_path'].format(ico,sub_ico)
    
    print(data_path)

    folder_to_save_model = config['logging']['folder_to_save_model'].format(ico,sub_ico)

    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    print(device)  

    ##############################
    ######     DATASET      ######
    ##############################

    print('')
    print('#'*35)
    print('Loading data <> ico {} - sub-ico {}'.format(ico,sub_ico))
    print('#'*35)
    print('')

    if str(config['data']['dataloader'])=='numpy':

        train_data = np.load(os.path.join(data_path,'train_data.npy'))
        train_label = np.load(os.path.join(data_path,'train_labels.npy'))

        print('training data: {}'.format(train_data.shape))

        val_data = np.load(os.path.join(data_path,'validation_data.npy'))
        val_label = np.load(os.path.join(data_path,'validation_labels.npy'))

        print('validation data: {}'.format(val_data.shape))

        train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                        torch.from_numpy(train_label).float())

        val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                        torch.from_numpy(val_label).float())

        train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                batch_size = bs,
                                                shuffle=True,
                                                num_workers=16)

        val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                                batch_size = bs_val,
                                                shuffle=False,
                                                num_workers=16)

        if testing:
            test_data = np.load(os.path.join(data_path,'test_data.npy'))
            test_label = np.load(os.path.join(data_path,'test_labels.npy'))
            print('testing data: {}'.format(test_data.shape))
            test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                            torch.from_numpy(test_label).float())
            test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                                    batch_size = bs_val,
                                                    shuffle=False,
                                                    num_workers=16)
    else:
        raise('not implemented yet')
    
              
    ##############################
    ######      LOGGING     ######
    ##############################

    # creating folders for logging. 

    folder_to_save_model = os.path.join(folder_to_save_model,config['data']['task'])
    try:
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))
    

    folder_to_save_model = os.path.join(folder_to_save_model,config['data']['data'])
    try:
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # folder time
    folder_to_save_model = os.path.join(folder_to_save_model,date)
    print(folder_to_save_model)
    if config['transformer']['dim'] == 192:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 384:
        folder_to_save_model = folder_to_save_model + '-small'
    elif config['transformer']['dim'] == 768:
        folder_to_save_model = folder_to_save_model + '-base'
    
    if config['training']['load_weights_imagenet']:
        folder_to_save_model = folder_to_save_model + '-imgnet'
    if config['training']['load_weights_ssl']:
        folder_to_save_model = folder_to_save_model + '-ssl'
        if config['training']['dataset_ssl']=='hcp':
            folder_to_save_model = folder_to_save_model + '-hcp'
        elif config['training']['dataset_ssl']=='dhcp':
            folder_to_save_model = folder_to_save_model + '-dhcp'
    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    try:
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))

    writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################

    print('')
    print('#'*30)
    print('Init model')
    print('#'*30)
    print('')

    model = SiT(dim=config['transformer']['dim'],
                    depth=config['transformer']['depth'],
                    heads=config['transformer']['heads'],
                    mlp_dim=config['transformer']['mlp_dim'],
                    pool=config['transformer']['pool'], 
                    num_patches=num_patches,
                    num_classes=config['transformer']['num_classes'],
                    num_channels=config['transformer']['num_channels'],
                    num_vertices=num_vertices,
                    dim_head=config['transformer']['dim_head'],
                    dropout=config['transformer']['dropout'],
                    emb_dropout=config['transformer']['emb_dropout'])
    
    if config['training']['load_weights_ssl']:

        print('Loading weights from self-supervision training')

        model.load_state_dict(torch.load(config['weights']['ssl_mpp'],map_location=device,strict=False)['model_state_dict'])
    
    if config['training']['load_weights_imagenet']:

        print('Loading weights from imagenet pretraining')

        model_trained = timm.create_model(config['weights']['imagenet'], pretrained=True)
        new_state_dict = load_weights_imagenet(model.state_dict(),model_trained.state_dict(),config['transformer']['depth'])
        model.load_state_dict(new_state_dict)
    
    if config['training']['finetuning']==False:
            print('freezing all layers except mlp head')

            for j, param in enumerate(model.parameters()):
                if j<136:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    model.to(device)

    if config['optimisation']['optimiser']=='Adam':
        print('using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        print('using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        print('using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('Optimiser not implemented yet')

    if not use_l1loss:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.L1Loss()


    if config['optimisation']['use_scheduler']:

        print('Using learning rate scheduler')

        if config['optimisation']['scheduler'] == 'StepLR':

            scheduler = StepLR(optimizer=optimizer,
                                step_size= config['StepLR']['stepsize'],
                                gamma= config['StepLR']['decay'])

            #if config['optimisation']['warmup']:
            #
            #    scheduler = GradualWarmupScheduler(optimizer,
            #                                       multiplier=1, 
            #                                       total_epoch=config['optimisation']['nbr_step_warmup'], 
            #                                       after_scheduler=scheduler)

        
        if config['optimisation']['scheduler'] == 'CosineDecay':

            scheduler = CosineAnnealingLR(optimizer,
                                        T_max = config['CosineDecay']['T_max'],
                                        eta_min=LR/10,
                                        )

            #if config['optimisation']['warmup']:
            #
            #    scheduler = GradualWarmupScheduler(optimizer,
            #                                       multiplier=1, 
            #                                       total_epoch=config['optimisation']['nbr_step_warmup'], 
            #                                       after_scheduler=scheduler)

        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                            factor=0.5,
                                            patience=250,
                                            cooldown=0,
                                            min_lr=0.000001
                                                )

            #if config['optimisation']['warmup']:
            #
            #    scheduler = GradualWarmupScheduler(optimizer,
            #                                       multiplier=1, 
            #                                       total_epoch=config['optimisation']['nbr_step_warmup'], 
            #                                       after_scheduler=scheduler)
     
    else:
        # to use warmup without fancy scheduler
        if config['optimisation']['warmup']:
            scheduler = StepLR(optimizer,
                                step_size=epochs)

            #scheduler = GradualWarmupScheduler(optimizer,
            #                                    multiplier=1, 
            #                                    total_epoch=config['optimisation']['nbr_step_warmup'], 
            #                                    after_scheduler=scheduler)
        

    best_mae = np.inf
    mae_val_epoch = np.inf
    running_val_loss = np.inf

    print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    print('Using {} criterion'.format(criterion))

    ##############################
    ######     TRAINING     ######
    ##############################

    print('')
    print('#'*30)
    print('Starting training')
    print('#'*30)
    print('')

    for epoch in range(epochs):

        running_loss = 0

        model.train()

        targets_ =  []
        preds_ = []

        for i, data in enumerate(train_loader):

            inputs, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs.squeeze(), targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            targets_.append(targets.cpu().numpy())
            preds_.append(outputs.reshape(-1).cpu().detach().numpy())

            writer.add_scalar('loss/train', loss.item(), epoch+1)

        mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))

        writer.add_scalar('mae/train',mae_epoch, epoch+1)

        if (epoch+1)%5==0:
            if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    if config['optimisation']['warmup']:
                        if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                            print('| Epoch - {} | Loss - {} | MAE - {} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4), scheduler.get_lr()[0] ))
                        else:
                            print('| Epoch - {} | Loss - {} | MAE - {} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
                    else:
                        print('| Epoch - {} | Loss - {} | MAE - {} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
                else:
                    print('| Epoch - {} | Loss - {} | MAE - {} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4), scheduler.get_lr()[0] ))
            else:
                print('| Epoch - {} | Loss - {} | MAE - {} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4), optimizer.param_groups[0]['lr']))

        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epoch==0:

            running_val_loss = 0

            model.eval()

            with torch.no_grad():

                targets_ = []
                preds_ = []

                for i, data in enumerate(val_loader):

                    inputs, targets = data[0].to(device), data[1].to(device)

                    outputs = model(inputs)

                    loss = criterion(outputs.squeeze(), targets)

                    running_val_loss += loss.item()

                    targets_.append(targets.cpu().numpy())
                    preds_.append(outputs.reshape(-1).cpu().numpy())


            writer.add_scalar('loss/val', running_val_loss, epoch+1)

            mae_val_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

            writer.add_scalar('mae/val',mae_val_epoch, epoch+1)

            print('| Validation | Epoch - {} | Loss - {} | MAE - {} |'.format(epoch+1, running_val_loss, mae_val_epoch ))

            if mae_val_epoch < best_mae:
                best_mae = mae_val_epoch
                best_epoch = epoch+1

                if testing:

                    model.eval()

                    print('starting testing')

                    with torch.no_grad():

                        targets_ = []
                        preds_ = []

                        for i, data in enumerate(test_loader):

                            inputs, targets = data[0].to(device), data[1].to(device)

                            outputs = model(inputs)

                            targets_.append(targets.cpu().numpy())
                            preds_.append(outputs.reshape(-1).cpu().numpy())

                        mae_test_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

                        print('| TESTING RESULTS | MAE - {} |'.format(mae_test_epoch))

                    df = pd.DataFrame()
                    df['preds'] = np.concatenate(preds_).reshape(-1)
                    df['targets'] = np.concatenate(targets_).reshape(-1)
                    df.to_csv(os.path.join(folder_to_save_model, 'preds_test.csv'))

                    config['logging']['folder_model_saved'] = folder_to_save_model
                    config['results'] = {}
                    config['results']['best_mae'] = float(best_mae)
                    config['results']['best_epoch'] = best_epoch
                    if testing:
                        config['results']['best_test_mae'] = float(mae_test_epoch)
                    config['results']['training_finished'] = False 

                    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                        yaml.dump(config, yaml_file)

                if config['training']['save_ckpt']:
                    print('saving model')
                    torch.save(model.state_dict(), os.path.join(folder_to_save_model,'checkpoint.pth'))
                    print('end saving model')

        if config['optimisation']['use_scheduler']:
            if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=mae_val_epoch)
                if config['optimisation']['warmup']:
                    if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                        writer.add_scalar('LR',scheduler.get_lr()[0], epoch+1 )
                    else:
                        writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch+1 )
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch+1 )
            else:
                scheduler.step()
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch+1 )
        else:
            if config['optimisation']['warmup']:
                scheduler.step()
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch+1 )
            else:
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch+1 )
    
    print('Final results: best model obtained at epoch {} - mean average error {}'.format(best_epoch,best_mae))

    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_mae'] = float(best_mae)
    config['results']['best_epoch'] = best_epoch
    if testing:
        config['results']['best_test_mae'] = float(mae_test_epoch)
    config['results']['training_finished'] = True 

    
    ##############################
    ######     TESTING      ######
    ##############################

    if testing:
        print('LOADING TESTING DATA: ICO {} - sub-res ICO {}'.format(ico,sub_ico))
        del train_data
        del val_data
        del model
        torch.cuda.empty_cache()

        test_model = SiT(dim=config['transformer']['dim'],
                    depth=config['transformer']['depth'],
                    heads=config['transformer']['heads'],
                    mlp_dim=config['transformer']['mlp_dim'],
                    pool=config['transformer']['pool'], 
                    num_patches=num_patches,
                    num_classes=config['transformer']['num_classes'],
                    num_channels=config['transformer']['num_channels'],
                    num_vertices=num_vertices,
                    dim_head=config['transformer']['dim_head'],
                    dropout=config['transformer']['dropout'],
                    emb_dropout=config['transformer']['emb_dropout'])


        print('loading model')
        test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'checkpoint.pth')))

        test_model.to(device)

        test_model.eval()

        print('starting testing')

        with torch.no_grad():

            targets_ = []
            preds_ = []

            for i, data in enumerate(test_loader):

                inputs, targets = data[0].to(device), data[1].to(device)

                outputs = test_model(inputs)

                targets_.append(targets.cpu().numpy())
                preds_.append(outputs.reshape(-1).cpu().numpy())

            mae_test_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

            print('| TESTING RESULTS | MAE - {} |'.format( mae_test_epoch ))

            config['results']['testing'] = float(mae_test_epoch)

            with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                yaml.dump(config, yaml_file)

            df = pd.DataFrame()
            df['preds'] = np.concatenate(preds_).reshape(-1)
            df['targets'] = np.concatenate(targets_).reshape(-1)
            df.to_csv(os.path.join(folder_to_save_model, 'preds_test.csv'))

    else:

        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


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
