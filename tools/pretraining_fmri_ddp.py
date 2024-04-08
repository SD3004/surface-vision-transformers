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
##torchrun --nproc_per_node=1 --nnodes=1  pretraining_fmri_ddp.py ../config/SiT/pretraining_fmri/hparams.yml

import os
import argparse
from statistics import mode
import yaml
import sys
import shutil

#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


sys.path.append('../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../models/')
sys.path.append('./workspace/fMRI_transformers/')
from tools.utils import logging_sit, get_data_path, get_dataloaders_distributed, get_dimensions, get_scheduler


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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()

def train(config):

    ###########################################
    ######     DISTRIBUTED TRAINING      ######
    ###########################################

    ## distributed training
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if rank ==0:
        print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
        print(rank, device, torch.cuda.device_count())

    #############################
    ######     CONFIG      ######
    #############################

    if rank ==0:
        
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

    #device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    if rank ==0:

        print('gpu: {}'.format(device))   
        print('dataset: {}'.format(dataset))  
        print('use confounds: {}'.format(use_confounds))
        print('task: {}'.format(task))  
        print('model: {}'.format(config['MODEL']))
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
        if rank ==0:
            print('')
            print('Mesh resolution - ico {}'.format(ico_mesh))
            print('Grid resolution - ico {}'.format(ico_grid))
            print('Number of patches - {}'.format(num_patches))
            print('Number of vertices - {}'.format(num_vertices))
            print('Reorder patches: {}'.format(config['mesh_resolution']['reorder']))
            print('')

    try:
        if str(dataloader)=='bold':
            train_loader, val_loader, train_sampler, val_sampler = get_dataloaders_distributed(config,data_path,world_size,rank)
        print('')
    except:
        raise("can't get dataloaders")


    ##############################
    ######      LOGGING     ######
    ##############################

    if rank ==0:

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


    if rank ==0: 
        writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################

    if rank ==0:

        print('')
        print('#'*30)
        print('######### Init model #########')
        print('#'*30)
        print('')

    if config['MODEL'] == 'sit' or config['MODEL'] == 'ms-sit':    

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

    if rank ==0:
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
    
    ssl = DDP(ssl.to(device), device_ids=[rank], find_unused_parameters=True)

    if rank==0:

        print('')   
        print('#'*60)
        print('Number of parameters pretraining pipeline : {:,}'.format(sum(p.numel() for p in ssl.parameters() if p.requires_grad)))
        print('Number of parameters encoder model: {:,}'.format(sum(p.numel() for p in ssl.module.encoder.parameters() if p.requires_grad)))
        print('Number of parameters decoder model: {:,}'.format(sum(p.numel() for p in ssl.module.decoder.parameters() if p.requires_grad)))
        print('#'*60)
        print('')

    if config['training']['restart']:
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
    else:
        iter_count= 0
        running_loss = 0 
        if rank==0:
            print('Training from scratch')

    #####################################
    #######     OPTIMISATION      #######
    #####################################

    if config['optimisation']['optimiser']=='Adam':
        if rank==0:
            print('using Adam optimiser')
        optimizer = optim.Adam(ssl.parameters(), lr=LR, 
                                                weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        if rank==0:
            print('using SGD optimiser')
        optimizer = optim.SGD(ssl.parameters(), lr=LR, 
                                                momentum=config['SGD']['momentum'],
                                                weight_decay=config['SGD']['weight_decay'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        if rank==0:
            print('using AdamW optimiser')
        optimizer = optim.AdamW(ssl.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')
    
    if config['training']['restart']:
        print('')
        print('#### LOADING OPTIMIZER STATE ####')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loading successfully')

    #import pdb;pdb.set_trace()

    ###################################
    #######     SCHEDULING      #######
    ###################################
    
    max_iterations = config['training']['iterations']
    log_training_it = config['training']['log_training_it']
    log_val_it = config['training']['log_val_it']

    scheduler = get_scheduler(config, max_iterations, optimizer)

    ##############################
    ######     TRAINING     ######
    ##############################

    if rank==0:
        print('')
        print('#'*30)
        print('###### Starting training #####')
        print('#'*30)
        print('')

    best_val_loss = 100000000000
    c_early_stop = 0
    epoch = 0
    
    while iter_count < max_iterations:

        train_sampler.set_epoch(epoch)

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
            mpp_loss_gpu = torch.tensor(mpp_loss.item(), device=device)
            dist.all_reduce(mpp_loss_gpu, op=dist.ReduceOp.SUM)
            mpp_loss_gpu =mpp_loss_gpu.item() / world_size
            if rank==0:
                scheduler, writer = tensorboard_log_pretrain_trainset(config, writer, scheduler, optimizer, mpp_loss_gpu ,iter_count+1)
            
            ##############################
            #########  LOG IT  ###########
            ##############################

            if (iter_count+1)%log_training_it==0:

                running_loss_gpu = torch.tensor(running_loss, device=device)
                dist.all_reduce(running_loss_gpu, op=dist.ReduceOp.SUM)
                running_loss_gpu =running_loss_gpu.item() / world_size
                running_loss_gpu_it = running_loss_gpu / (iter_count+1) 

                if rank==0:
                    log_pretrain(config, optimizer, scheduler, iter_count+1, running_loss_gpu_it)

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
                val_sampler.set_epoch(epoch)

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
                    
                loss_pretrain_val_epoch = running_val_loss/(i+1)
                loss_pretrain_val_epoch_gpu = torch.tensor(loss_pretrain_val_epoch,device=device)
                dist.all_reduce(loss_pretrain_val_epoch_gpu, op=dist.ReduceOp.SUM)
                loss_pretrain_val_epoch_gpu = loss_pretrain_val_epoch_gpu.item() / world_size

                if rank==0:

                    writer = tensorboard_log_pretrain_valset(writer, loss_pretrain_val_epoch_gpu, iter_count+1)

                    if loss_pretrain_val_epoch_gpu < best_val_loss:
                        best_val_loss = loss_pretrain_val_epoch_gpu
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
                                                    running_loss_gpu_it,
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
        
        epoch += 1

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

    if rank==0:
        
        print('Saving final checkpoint...')

        torch.save({'epoch':iter_count+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':running_loss_gpu_it,
                    },
                    os.path.join(folder_to_save_model,'encoder-final.pt'))

        torch.save({'epoch':iter_count+1,
                    'model_state_dict': ssl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':running_loss_gpu_it,
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

    # Call training
    train(config)


