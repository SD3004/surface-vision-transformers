# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-04-07 15:51:18
#
# Created on Wed Oct 20 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import nibabel as nb
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import subprocess

from einops.layers.torch import Rearrange

from tools.dataloader import loader_metrics, loader_metrics_segmentation

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from warmup_scheduler import GradualWarmupScheduler


def get_data_path(config):

    dataloader = config['data']['dataloader']
    dataset = config['data']['dataset']
    task = config['data']['task']
    configuration = config['data']['configuration']
    modality = config['data']['modality']
    sampling = config['mesh_resolution']['sampling']

    if str(dataloader) == 'metrics':
        if dataset == 'dHCP':
            data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/ico_6_{}'.format(sampling),'base','regression_{}_space_features'.format(configuration))
        elif dataset == 'UKB':
            if modality == 'cortical_metrics':
                if task == 'scan_age_msmall' or task == 'sex_msmall':
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_msmall/')
                else:
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample/')
            elif modality == 'fMRI':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_timeseries/clean_vn/')
        elif dataset == 'HCP':
            if modality == 'cortical_metrics':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'merged_metrics/ico6_{}_{}'.format(sampling,configuration), modality)
            elif modality == 'memory_task':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'merged_metrics/ico6_{}_{}'.format(sampling,configuration), modality)
    else:
        raise('not implemented yet')
    
    return data_path

def get_data_path_segmentation(config):

    dataloader = config['data']['dataloader']
    dataset = config['data']['dataset']
    task = config['data']['task']
    modality = config['data']['modality']

    if str(dataloader) == 'metrics':
        if dataset == 'UKB':
            if modality == 'cortical_metrics':
                if task == 'segmentation_msmall':
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_msmall/')
                    labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps')
                elif task == 'segmentation':
                    if config['data']['masking_preprocess']:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_{}_mask/'.format(config['data']['masking_preprocess']))
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps_{}_mask'.format(config['data']['masking_preprocess']))  
                    else:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample/')
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps')  
                                
        elif dataset == 'MindBoggle':
            if modality == 'cortical_metrics':
                if task == 'segmentation':
                    if config['data']['masking_preprocess']:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'mindboggle_merged_metrics_{}_mask'.format(config['data']['masking_preprocess']))
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'mindboggle_resample_labels_ico6_{}_mask'.format(config['data']['masking_preprocess'])) 
                    else:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'mindboggle_merged_metrics')
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'mindboggle_resample_labels_ico6') 
                          
    else:
        raise('not implemented yet')
    
    return data_path, labels_path


def get_dataloaders(config, 
                    data_path):

    dataloader = config['data']['dataloader']
    sampler = config['training']['sampler']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    modality = config['data']['modality']

    if str(dataloader)=='metrics':
        if str(modality) == 'cortical_metrics' or str(modality) == 'memory_task':
            train_loader, val_loader, test_loader = loader_metrics(data_path,sampler,config)

    else:
        raise('not implemented yet')
    
    return train_loader, val_loader, test_loader



def get_dataloaders_segmentation(config, 
                    data_path,
                    labels_path,):

    dataloader = config['data']['dataloader']
    sampler = config['training']['sampler']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    modality = config['data']['modality']

    if str(dataloader)=='metrics':
        if str(modality) == 'cortical_metrics' or str(modality) == 'memory_task':
            train_loader, val_loader, test_loader = loader_metrics_segmentation(data_path,labels_path,sampler,config)
        else:
            raise('not implemented yet')
    else:
        raise('not implemented yet')
    
    return train_loader, val_loader, test_loader

def get_dataloaders_cv(config, 
                        data_path,
                        k):

    sampler = config['training']['sampler']

    train_loader, test_loader = loader_metrics(data_path,
                                                sampler,
                                                config,
                                                split_cv=k)

    return train_loader, test_loader
    
    
def get_dimensions(config):

    modality = config['data']['modality']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    if config['MODEL'] in ['sit','ms-sit']:    
        channels = config['transformer']['channels']
    elif config['MODEL']== 'spherical-unet':
        channels = config['spherical-unet']['channels']
    num_channels = len(channels)

    if config['MODEL'] in ['sit','ms-sit']:    
        
        T = num_channels
        N = num_patches
        
        V = num_vertices
           

        use_bottleneck = False
        bottleneck_dropout = 0.0

        print('Number of channels {}; Number of patches {}; Number of vertices {}'.format(T, N, V))
        print('Using bottleneck {}; Dropout bottleneck {}'.format(use_bottleneck,bottleneck_dropout))
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout

def get_scheduler(config, nbr_iteration_per_epoch ,optimizer):

    epochs = config['training']['epochs']

    if config['optimisation']['use_scheduler']:

        print('Using learning rate scheduler')

        if config['optimisation']['scheduler'] == 'StepLR':

            scheduler = StepLR(optimizer=optimizer,
                                step_size= config['StepLR']['stepsize'],
                                gamma= config['StepLR']['decay'])
        
        elif config['optimisation']['scheduler'] == 'CosineDecay':

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                    T_max = config['CosineDecay']['T_max'],
                                                                    eta_min= config['CosineDecay']['eta_min'],
                                                                    )

        elif config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                            mode='max',
                                            factor=0.5,
                                            patience=2,
                                            cooldown=0,
                                            min_lr=0.0000001
                                                )

        if config['optimisation']['warmup']:

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
     
    else:
        # to use warmup without fancy scheduler
        if config['optimisation']['warmup']:
            scheduler = StepLR(optimizer,
                                step_size=epochs*nbr_iteration_per_epoch)

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
        else:

            return None
            
    return scheduler



def load_weights_imagenet(state_dict,state_dict_imagenet,nb_layers):

    state_dict['mlp_head.0.weight'] = state_dict_imagenet['norm.weight'].data
    state_dict['mlp_head.0.bias'] = state_dict_imagenet['norm.bias'].data

    # transformer blocks
    for i in range(nb_layers):
        state_dict['transformer.layers.{}.0.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm1.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm2.bias'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_qkv.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.qkv.weight'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_out.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.fn.to_out.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.3.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.3.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.bias'.format(i)].data

    return state_dict



def save_gifti(data, filename):
    gifti_file = nb.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(data))
    nb.save(gifti_file,filename)

def save_label_UKB(data, filename):
    label =nb.load('/home/sd20/data/UKB/metrics/resample_segmentation_maps/1033131.L.aparc.ico6_fs_LR.label.gii')
    label.darrays[0].data = data
    nb.save(label,filename)

def save_label_MindBoggle(data, filename):
    label =nb.load('/home/sd20/data/MindBoggle/mindboggle_resample_labels_ico6/lh.labels.HLN-12-5.ico6.DKT31.manual.label.gii')
    label.darrays[0].data = data
    nb.save(label,filename)


def logging_sit(config, pretraining=False):

    if pretraining:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],'pretraining',config['data']['task'],config['SSL'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    else:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['modality'],config['data']['task'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)

    if config['transformer']['dim'] == 192:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 384:
        folder_to_save_model = folder_to_save_model + '-small'
    elif config['transformer']['dim'] == 768:
        folder_to_save_model = folder_to_save_model + '-base'
    elif config['transformer']['dim'] == 96:
        folder_to_save_model = folder_to_save_model + '-very-tiny'
    elif config['transformer']['dim'] == 48:
        folder_to_save_model = folder_to_save_model + '-ultra-tiny'
    

    if config['training']['init_weights']!=False:
        folder_to_save_model = folder_to_save_model + '-'+config['training']['init_weights']

    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    return folder_to_save_model

def logging_ms_sit(config, pretraining=False):

    if pretraining:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],'pretraining',config['data']['task'],config['SSL'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    else:
        if config['data']['task'] =='segmentation':
            folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],'{}_mask'.format(config['data']['masking_preprocess']),config['mesh_resolution']['ico_grid'],config['data']['configuration'])
        else:
            folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)

    if config['transformer']['dim'] == 96:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 48:
        folder_to_save_model = folder_to_save_model + '-very-tiny'
    

    if config['training']['init_weights']!=False:
        folder_to_save_model = folder_to_save_model + '-'+config['training']['init_weights']

    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    return folder_to_save_model


def logging_spherical_unet(config):

    folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],config['data']['configuration'])
    
    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)

    return folder_to_save_model



def plot_regression_results_UKB(predictions, targets,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))
    
    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([45,80],[45,80],color='r')
    plt.savefig(path_to_save)


def save_segmentation_results_UKB(predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    val_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/UKB/cortical_metrics/segmentation/half/val.csv').ids
    for i, id in enumerate(val_ids):
        save_label_UKB(predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))

def save_segmentation_results_MindBoggle(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass
    
    if config['data']['hemi_part']=='all':
        val_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/val.csv').ids
    elif config['data']['hemi_part']=='left':
        val_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/val_L.csv').ids
    elif config['data']['hemi_part']=='right':
        val_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/val_R.csv').ids

    for i, id in enumerate(val_ids):
        save_label_MindBoggle(predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))

def save_segmentation_results_UKB_test(predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    test_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/UKB/cortical_metrics/segmentation/half/test.csv').ids
    for i, id in enumerate(test_ids):
        save_label_UKB(predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))

def save_segmentation_results_MindBoggle_test(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass
    
    if config['data']['hemi_part']=='all':
        test_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/test.csv').ids
    elif config['data']['hemi_part']=='left':
        test_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/test_L.csv').ids
    elif config['data']['hemi_part']=='right':
        test_ids = pd.read_csv('/home/sd20/workspace/transformers/labels/MindBoggle/cortical_metrics/segmentation/half/test_R.csv').ids

    for i, id in enumerate(test_ids):
        save_label_MindBoggle(predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))


def plot_regression_results_dHCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([25,48],[25,48],color='r')
    plt.plot([20,40],[20,40],color='r')
    plt.savefig(path_to_save)


def plot_regression_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([20,40],[20,40],color='r')
    plt.savefig(path_to_save)


def plot_handedness_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([-100,100],[-100,100],color='r')
    plt.savefig(path_to_save)

def plot_iq_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([5,27],[5,27],color='r')
    plt.savefig(path_to_save)


def save_reconstruction_mae(reconstructed_batch,
                            reconstructed_batch_unmasked,
                            inputs, 
                            num_patches,
                            num_vertices,
                            ico_grid,
                            num_channels,
                            masked_indices,
                            unmasked_indices,
                            epoch,
                            folder_to_save_model,
                            split,
                            ):

    try:
        os.makedirs(os.path.join(folder_to_save_model, 'reconstruction', '{}'.format(split)),exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        pass

    indices = pd.read_csv('../patch_extraction/msm/triangle_indices_ico_6_sub_ico_{}.csv'.format(ico_grid))

    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    new_inputs = np.transpose(inputs.cpu().numpy(),(0,2,1,3))

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(folder_to_save_model,'reconstruction','{}'.format(split), 'original_sphere_{}.shape.gii'.format(epoch)))

    B, num_masked_patch, V = reconstructed_batch.shape
    rearrange_layer_masked = Rearrange('b m (v c) -> b c m v', b=B, m=num_masked_patch, c=num_channels, v=num_vertices)

    B, num_unmasked_patch, V = reconstructed_batch_unmasked.shape
    rearrange_layer_unmasked = Rearrange('b m (v c) -> b c m v', b=B, m=num_unmasked_patch, c=num_channels, v=num_vertices)

    batch = rearrange_layer_masked(reconstructed_batch).cpu().numpy()
    batch_unmasked = rearrange_layer_unmasked(reconstructed_batch_unmasked).cpu().numpy()

    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        #import pdb;pdb.set_trace()
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch[0,:,ind,:].transpose()
        elif i in unmasked_indices:
            ind =  (unmasked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch_unmasked[0,:,ind,:].transpose()
        else:
            print('issue with indices: {}'.format(i))

    #import pdb;pdb.set_trace()
    save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,'reconstruction', '{}'.format(split), 'reconstructed_sphere_{}.shape.gii'.format(epoch)))

    sphere_patched = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        #import pdb;pdb.set_trace()
        if i in masked_indices:
            #import pdb;pdb.set_trace()
            sphere_patched[indices_to_extract,:] = 0
        else:
            #import pdb;pdb.set_trace()
            sphere_patched[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    #import pdb;pdb.set_trace()
    save_gifti(sphere_patched, os.path.join(folder_to_save_model,'reconstruction', '{}'.format(split), 'sphere_patched_{}.shape.gii'.format(epoch)))

def save_reconstruction_mae_test(reconstructed_batch,
                                reconstructed_batch_unmasked,
                                inputs, 
                                num_patches,
                                num_vertices,
                                ico_grid,
                                num_channels,
                                masked_indices,
                                unmasked_indices,
                                folder_to_save_model,
                                id,
                                split,
                                hemi,
                                ):

    try:
        os.makedirs(os.path.join(folder_to_save_model, 'reconstructions_test_time', '{}'.format(split)),exist_ok=False)
        print('Creating folder: {}'.format(os.path.join(folder_to_save_model, 'reconstructions_test_time', '{}'.format(split))))
    except OSError:
        pass

    indices = pd.read_csv('../patch_extraction/msm/triangle_indices_ico_6_sub_ico_{}.csv'.format(ico_grid))

    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    new_inputs = np.transpose(inputs.cpu().numpy(),(0,2,1,3))

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(folder_to_save_model,'reconstructions_test_time','{}'.format(split), '{}_{}.sphere.shape.gii'.format(id,hemi)))

    p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.sphere.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
    p1.wait()

    B, num_masked_patch, V = reconstructed_batch.shape
    rearrange_layer_masked = Rearrange('b m (v c) -> b c m v', b=B, m=num_masked_patch, c=num_channels, v=num_vertices)

    B, num_unmasked_patch, V = reconstructed_batch_unmasked.shape
    rearrange_layer_unmasked = Rearrange('b m (v c) -> b c m v', b=B, m=num_unmasked_patch, c=num_channels, v=num_vertices)

    batch = rearrange_layer_masked(reconstructed_batch).cpu().numpy()
    batch_unmasked = rearrange_layer_unmasked(reconstructed_batch_unmasked).cpu().numpy()

    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch[0,:,ind,:].transpose()
        elif i in unmasked_indices:
            ind =  (unmasked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch_unmasked[0,:,ind,:].transpose()
        else:
            print('issue with indices: {}'.format(i))

    #import pdb;pdb.set_trace()
    save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split),'{}_{}.reconstruction.shape.gii'.format(id,hemi)))

    p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.reconstruction.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
    p1.wait()

    reconstructed_sphere_mask_only = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere_mask_only[indices_to_extract,:] = batch[0,:,ind,:].transpose()

    save_gifti(reconstructed_sphere_mask_only, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split),'{}_{}.reconstruction_mask_only.shape.gii'.format(id,hemi)))

    p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.reconstruction_mask_only.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
    p1.wait()

    sphere_patched = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            sphere_patched[indices_to_extract,:] = 0
        else:
            sphere_patched[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(sphere_patched, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split), '{}_{}.sphere_masked.shape.gii'.format(id,hemi)))

    p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.sphere_masked.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
    p1.wait()

