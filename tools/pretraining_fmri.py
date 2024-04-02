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
import shutil
import glob

#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


sys.path.append('../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../models/')
sys.path.append('/nfs/home/sdahan/workspace/sMAE/')
from tools.utils import logging_sit, get_data_path, get_dataloaders, get_dimensions, get_scheduler


import torch
import torch.optim as optim

from models.sit import SiT
from models.mpp import masked_patch_pretraining
from models.mae import MAE
from models.smae import sMAE
from models.vsmae import vsMAE

from einops import rearrange

from torch.utils.tensorboard import SummaryWriter
from tools.log import tensorboard_log_pretrain_trainset, log_pretrain, save_reconstruction_pretain, \
                        tensorboard_log_pretrain_valset, saving_ckpt_pretrain, save_reconstruction_pretrain_fmri, \
                        save_reconstruction_pretrain_fmri_valset


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
    configuration = config['data']['configuration']
    hemi = config['data']['hemi']
    task = config['data']['task'] 

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    use_confounds = config['training']['use_confounds']
    early_stopping = config['training']['early_stopping']
    dataloader = config['data']['dataloader']
    restart = config['training']['restart']

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
    print('model: {}'.format(config['MODEL']))
    print('configuration: {}'.format(configuration))  
    print('data path: {}'.format(data_path))
    if restart:
        path_to_previous_ckpt = config['training']['path_from_ckpt']
        try:
            assert os.path.exists(path_to_previous_ckpt)
            print('path to checkpoint exist')
        except:
            raise NotADirectoryError
        print('')
        print('### RESTARTING TRAINING FROM: ###')
        print(path_to_previous_ckpt)
        print('')

    ##############################
    ######     DATASET      ######
    ##############################

    if config['MODEL'] == 'sit' or config['MODEL'] == 'ms-sit':
        print('')
        print('Mesh resolution - ico {}'.format(ico_mesh))
        print('Grid resolution - ico {}'.format(ico_grid))
        print('Number of patches - {}'.format(num_patches))
        print('Number of vertices - {}'.format(num_vertices))
        print('Reorder patches: {}'.format(config['mesh_resolution']['reorder']))
        print('')

    try:
        if str(dataloader)=='metrics':
            train_loader, val_loader, test_loader = get_dataloaders(config,data_path)
        elif str(dataloader)=='bold':
            train_loader, val_loader = get_dataloaders(config,data_path)
        print('')
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

    if not config['SERVER']:

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
        
    else:
        print('not creating a saving folder path')
        folder_to_save_model = '/'.join(config['CONFIG_PATH'].split('/')[:-1])
        print(folder_to_save_model)

    #Continue training. Training has been killed ? 
    pt_files = glob.glob(os.path.join(folder_to_save_model, '*.pt'))

    # Check if the list of files is not empty
    if pt_files:
        print("There are .pt files in the folder.")
        for file in pt_files:
            print(file)
        continue_training = True

        ## log restart config file:
        config['RESTART_TRAINING_ID'] = int(config['RESTART_TRAINING_ID']) + 1
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)
    else:
        print("There are no .pt files in the folder.")
        continue_training = False

    # cant' have both restarting training and continue training 
    #assert( (config['training']['restart'] and not continue_training) or (not config['training']['restart'] and continue_training))

    #tensorboard
    if restart:
        #copy the previous tb event
        for file in os.listdir(path_to_previous_ckpt):
            if file.startswith("events.out.tfevents"):
                event_file_path = os.path.join(path_to_previous_ckpt, file)
                
                # Copy the event file to the new directory
                shutil.copy(event_file_path, folder_to_save_model)
                print(f"Copied {file} to {folder_to_save_model}")

        for file in os.listdir(path_to_previous_ckpt):
            if file.startswith("hparams.yml"):
                event_file_path = os.path.join(path_to_previous_ckpt, 'hparams.yml')
                
                # Copy the event file to the new directory
                shutil.copy(event_file_path, os.path.join(folder_to_save_model,'hparams_old.yml'))
                print(f"Copied {file} to {folder_to_save_model}")


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
    
    print('Dimensions: {}, {},{}'.format(T,N,V))

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
                        trainable_pos_emb=config['transformer']['trainable_pos_emb'],
                        no_class_token_emb = config['transformer']['no_class_token_emb'],)

    model.to(device)

    print('Number of parameters encoder: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    ### Loading weights for SiT Encoder ###

    if continue_training:

        # in that case there will be an encoder ckpt in the folder
        print('##### Loading BEST checkpoint for SiT encoder model #####')
        checkpoint = torch.load(os.path.join(folder_to_save_model,'encoder-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True) 
        print('Loaded the SiT encoder model successfully')

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
                    decoder_dim=config['pretraining_mae']['decoder_dim'],
                    masking_ratio=config['pretraining_mae']['mask_prob'],
                    decoder_depth=config['pretraining_mae']['decoder_depth'],
                    decoder_heads = config['pretraining_mae']['decoder_heads'],
                    decoder_dim_head = config['pretraining_mae']['decoder_dim_head'],
                    use_pos_embedding_decoder=config['pretraining_mae']['use_pos_embedding_decoder'],
                    use_all_patch_loss= config['pretraining_mae']['use_all_patch_loss'],
                    mask=config['data']['masking'],
                    dataset=dataset,
                    configuration = configuration,
                    sampling=sampling,
                    sub_ico=ico_grid,
                    num_channels=num_channels,
                    weights_init=config['pretraining_mae']['init_weights'],
                    path_to_template=config['data']['path_to_template'],
                    path_to_workdir= config['data']['path_to_workdir'])
        
    elif config['SSL'] == 'smae':

        print('Pretrain using Masked AutoEncoder')
        ssl = sMAE(encoder=model, 
                    decoder_dim=config['pretraining_smae']['decoder_dim'],
                    masking_ratio=config['pretraining_smae']['mask_prob'],
                    decoder_depth=config['pretraining_smae']['decoder_depth'],
                    decoder_heads = config['pretraining_smae']['decoder_heads'],
                    decoder_dim_head = config['pretraining_smae']['decoder_dim_head'],
                    dataset=dataset,
                    configuration = configuration,
                    num_channels=num_channels,
                    weights_init=config['pretraining_smae']['init_weights'],
                    no_class_emb_decoder=config['pretraining_smae']['no_class_emb_decoder'],
                    mask=config['data']['masking'],
                    path_to_template=config['data']['path_to_template'],
                    path_to_workdir= config['data']['path_to_workdir'],
                    sampling=sampling,
                    sub_ico=ico_grid,)
    
    elif config['SSL'] == 'vsmae':

        print('Pretrain using Vision Surface Masked AutoEncoder')
        ssl = vsMAE(encoder=model, 
                    decoder_dim=config['pretraining_vsmae']['decoder_dim'],
                    masking_ratio=config['pretraining_vsmae']['mask_prob'],
                    decoder_depth=config['pretraining_vsmae']['decoder_depth'],
                    decoder_heads = config['pretraining_vsmae']['decoder_heads'],
                    decoder_dim_head = config['pretraining_vsmae']['decoder_dim_head'],
                    dataset=dataset,
                    configuration = configuration,
                    num_channels=num_channels,
                    weights_init=config['pretraining_vsmae']['init_weights'],
                    no_pos_emb_class_token_decoder=config['pretraining_vsmae']['no_pos_emb_class_token_decoder'],
                    mask=config['data']['masking'],
                    path_to_template=config['data']['path_to_template'],
                    path_to_workdir= config['data']['path_to_workdir'],
                    sampling=sampling,
                    sub_ico=ico_grid,
                    masking_type=config['pretraining_vsmae']['masking_type'],
                    temporal_rep = config['fMRI']['temporal_rep'],
                    nbr_frames = config['fMRI']['nbr_frames'],
                    loss=config['pretraining_vsmae']['loss'],
                    mask_loss=config['pretraining_vsmae']['mask_loss'])
        
        print('Masking type: {}'.format(config['pretraining_vsmae']['masking_type']))  
    else:
        raise('not implemented yet')  
    
    ssl.to(device)

    print('Number of parameters pretraining pipeline : {:,}'.format(sum(p.numel() for p in ssl.parameters() if p.requires_grad)))
    print('')

    if config['training']['restart'] and (not continue_training):

        if os.path.exists(os.path.join(config['training']['path_from_ckpt'],'encoder-decoder-final.pt')):
            print('##### Loading FINAL checkpoint #####')
            checkpoint = torch.load(os.path.join(config['training']['path_from_ckpt'],'encoder-decoder-final.pt'))
        else:
            print('##### Loading BEST checkpoint #####')
            checkpoint = torch.load(os.path.join(config['training']['path_from_ckpt'],'encoder-decoder-best.pt'))

        ssl.load_state_dict(checkpoint['model_state_dict'],strict=True) 
        iter_count = checkpoint['epoch']
        running_loss = checkpoint['loss'] * (iter_count-1)
        print('Starting training from iteration  {}'.format(iter_count))

    elif continue_training:

        print('##### Loading BEST checkpoint for encoder-decoder  model #####')
        checkpoint = torch.load(os.path.join(folder_to_save_model,'encoder-decoder-best.pt'))
        ssl.load_state_dict(checkpoint['model_state_dict'],strict=True) 

    else: 
        print('Training from scratch')

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
    
    if config['training']['restart'] and (not continue_training):
        print('')
        print('#### LOADING OPTIMIZER STATE ####')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loading successfully')

    elif continue_training:
        print('')
        print('#### LOADING OPTIMIZER STATE ####')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loading successfully')
        iter_count = checkpoint['epoch']
        running_loss = checkpoint['loss'] * (iter_count-1)
        print('Starting training from iteration  {}'.format(iter_count))
    else:
        iter_count= 0
        running_loss = 0 
        print('Training from scratch')
    
    ###################################
    #######     SCHEDULING      #######
    ###################################
    
    max_iterations = config['training']['iterations']
    log_training_it = config['training']['log_training_it']
    log_val_it = config['training']['log_val_it']

    scheduler = get_scheduler(config, max_iterations, optimizer)

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
    
    while iter_count < max_iterations:

        for i, data in enumerate(train_loader):

            ssl.train()
            
            optimizer.zero_grad()

            if task != 'None':
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs = data.to(device)

            if config['training']['runtime']:
                B,b,t,n,v = inputs.shape
                inputs = rearrange(inputs, 'b t c n v -> (b t) c n v') 
            else:
                B,t,n,v = inputs.shape
                
            if use_confounds:

                confounds = labels[:,1]
                if config['SSL'] == 'vsmae':
                    mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices= ssl(inputs,confounds)
            else:
                if config['SSL'] == 'vsmae':
                    mpp_loss, reconstructed_batch_token_masked, reconstructed_batch_token_not_masked, ids_tokens_masked, ids_tokens_not_masked = ssl(inputs)
                elif config['SSL'] == 'mpp':
                    mpp_loss, _ = ssl(inputs)


            mpp_loss.backward()
            optimizer.step()

            running_loss += mpp_loss.item()
            #import pdb;pdb.set_trace()

            #tensorboard log train
            scheduler, writer = tensorboard_log_pretrain_trainset(config, writer, scheduler, optimizer, mpp_loss.item(),iter_count+1)
            
            ##############################
            #########  LOG IT  ###########
            ##############################


            if (iter_count+1)%log_training_it==0:

                #loss_pretrain_it = running_loss / (iter_count+1) #before 26.02.2024
                loss_pretrain_it = mpp_loss.item()

                log_pretrain(config, optimizer, scheduler, iter_count+1, loss_pretrain_it)

                if (iter_count+1)%1000==0:

                    save_reconstruction_pretrain_fmri(config,
                                                    reconstructed_batch_token_masked[:1],
                                                    reconstructed_batch_token_not_masked[:1],
                                                    inputs[:1],
                                                    ids_tokens_masked[:1],
                                                    ids_tokens_not_masked[:1],
                                                    iter_count+1,
                                                    folder_to_save_model,)

            ##############################
            ######    VALIDATION    ######
            ##############################

            if (iter_count+1)%log_val_it==0:

                running_val_loss = 0
                ssl.eval()

                with torch.no_grad():

                    for i, data in enumerate(val_loader):

                        if task != 'None':
                            inputs, labels = data[0].to(device), data[1].to(device)
                        else:
                            inputs = data.to(device)

                        if  config['SSL'] == 'vsmae':
                            mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices = ssl(inputs)
                        elif config['SSL'] == 'mpp':
                            mpp_loss, _ = ssl(inputs)

                        running_val_loss += mpp_loss.item()
                    
                loss_pretrain_val_epoch = running_val_loss /(i+1)

                writer = tensorboard_log_pretrain_valset(writer, loss_pretrain_val_epoch, iter_count+1)

                if loss_pretrain_val_epoch < best_val_loss:
                    best_val_loss = loss_pretrain_val_epoch
                    c_early_stop = 0

                    if (config['SSL'] == 'vsmae' and config['pretraining_vsmae']['save_reconstruction']):
                        for i, data in enumerate(val_loader):
                            if i<3:
                                if task != 'None':
                                    inputs, labels = data[0].to(device), data[1].to(device)
                                else:
                                    inputs = data.to(device)

                                mpp_loss, reconstructed_batch, reconstructed_batch_unmasked, masked_indices, unmasked_indices = ssl(inputs)

                                if (iter_count+1)%1000==0:

                                    save_reconstruction_pretrain_fmri_valset(config,
                                                                reconstructed_batch,
                                                                reconstructed_batch_unmasked,
                                                                inputs,
                                                                masked_indices, 
                                                                unmasked_indices,
                                                                iter_count+1,
                                                                folder_to_save_model,
                                                                id=i,
                                                                )
                    

                    config = saving_ckpt_pretrain(config,
                                                  iter_count+1,
                                                loss_pretrain_it,
                                                loss_pretrain_val_epoch,
                                                folder_to_save_model,
                                                model,
                                                ssl,
                                                optimizer)
        
                    
                elif early_stopping:
                    c_early_stop += 1

                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                    yaml.dump(config, yaml_file)
            
            iter_count += 1
            if iter_count >= max_iterations:
                break
            
            if early_stopping and (c_early_stop>=early_stopping):
                print('stop training - early stopping')
                break
    
            ##################################
            ######   UPDATE SCHEDULER  #######
            ##################################
            
            if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(metrics=loss_pretrain_val_epoch)

    print('Training is finished!')

    if early_stopping and (c_early_stop>=early_stopping):
        config['results']['training_finished'] = 'early stopping' 
    else:
        config['results']['training_finished'] = True 

    config['results']['final_loss'] = loss_pretrain_it

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    
    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################
    
    print('Saving final checkpoint...')

    torch.save({'epoch':iter_count+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_it,
                },
                os.path.join(folder_to_save_model,'encoder-final.pt'))

    torch.save({'epoch':iter_count+1,
                'model_state_dict': ssl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_it,
                },
                os.path.join(folder_to_save_model,'encoder-decoder-final.pt'))
    
    print('Checkpoint saved!')

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
    
    config['CONFIG_PATH'] = args.config

    # Call training
    train(config)


