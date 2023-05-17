import os
import sys

import numpy as np
import nibabel as nb
import pandas as pd

sys.path.append('./')
sys.path.append('./tools')

from tools.datasets import dataset_cortical_surfaces, dataset_cortical_surfaces_segmentation

from tools.samplers import new_sampler_HCP_fluid_intelligence,sampler_preterm_birth_age, sampler_preterm_scan_age, sampler_sex_classification, sampler_UKB_scan_age, sampler_HCP_fluid_intelligence

import torch

def loader_metrics(data_path,
                    sampler,
                    config,
                    split_cv=-1):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_cortical_surfaces(config=config,
                                                data_path=data_path,
                                                split='train',
                                                split_cv=split_cv)


    #####################################
    ###############  dHCP  ##############
    #####################################
    if sampler and config['data']['dataset']=='dHCP' :
        
        if config['data']['task']=='birth_age' :
            print('Sampler: dHCP preterm for birth_age')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_preterm_birth_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
        elif config['data']['task']=='scan_age':
            print('Sampler: dHCP preterm for scan_age')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_preterm_scan_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
            
        elif config['data']['task']=='sex':
            print('Sampler: dHCP sex classification')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_sex_classification(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
                                            
    #####################################
    ###############  UKB   ##############
    #####################################
    elif sampler and config['data']['dataset']=='UKB':
    
        if config['data']['task']=='sex' or config['data']['task']=='sex_msmall':
            print('Sampler: UKB sex classification')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_sex_classification(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)

        elif config['data']['task']=='scan_age':
            print('Sampler: UKB scan age regression')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_UKB_scan_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)

    #####################################
    ###############  HCP   ##############
    #####################################
    elif sampler and config['data']['dataset']=='HCP':
        
        if config['training']['use_cross_validation']: 
        
            if config['data']['task']=='iq':

                print('Sampler: HCP fluid intelligence regression')

                train_labels = pd.read_csv('{}/labels/{}/{}/cv_5/{}/{}/train{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['modality'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],
                                                                                            split_cv)).labels.to_numpy()
                sampler = new_sampler_HCP_fluid_intelligence(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
            if config['data']['task']=='sex':

                print('Sampler: HCP fluid intelligence regression')

                train_labels = pd.read_csv('{}/labels/{}/{}/cv_5/{}/{}/train{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['modality'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],
                                                                                            split_cv)).labels.to_numpy()
                sampler = sampler_sex_classification(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
            
            
        # non cross validation
        else:
            if config['data']['task']=='sex':

                print('Sampler: HCP sex classification')

                train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],'train')).labels.to_numpy()
                sampler = sampler_sex_classification(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
    else:
        print('not using sampler...')
        print('shuffling == {}'.format(not config['RECONSTRUCTION']))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = config['training']['bs'],
                                                    shuffle=(not config['RECONSTRUCTION']),
                                                    num_workers=32)

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    if split_cv == -1:

        val_dataset = dataset_cortical_surfaces(data_path=data_path,
                                                    config=config,
                                                    split='val',
                                                    split_cv=split_cv)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['training']['bs_val'],
                                                shuffle=False,
                                                num_workers=32)
                

    ###############################################################
    #####################    TESTING DATA     #####################
    ###############################################################
            

    test_dataset = dataset_cortical_surfaces(data_path=data_path,
                                            config=config,
                                            split='test',
                                            split_cv=split_cv)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False, 
                                            num_workers=32)
    
    train_dataset.logging()
    
    if split_cv ==-1:

        print('')
        print('#'*30)
        print('############ Data ############')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Validation data: {}'.format(len(val_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, val_loader, test_loader
    
    else:

        print('')
        print('#'*30)
        print('########### Data ###########')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, test_loader



def loader_metrics_segmentation(data_path,
                                labels_path,
                                sampler,
                                config,
                                split_cv=-1):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_cortical_surfaces_segmentation(config=config,
                                                            data_path=data_path,
                                                            labels_path=labels_path,
                                                            split='train',
                                                            split_cv=split_cv)

    #####################################
    ###############  UKB   ##############
    #####################################


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = config['training']['bs'],
                                                shuffle = True, 
                                                num_workers=32)

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    if split_cv == -1:

        val_dataset = dataset_cortical_surfaces_segmentation(data_path=data_path,
                                                    config=config,
                                                    labels_path=labels_path,
                                                    split='val',
                                                    split_cv=split_cv)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['training']['bs_val'],
                                                shuffle=False,
                                                num_workers=32)
                

    ###############################################################
    #####################    TESTING DATA     #####################
    ###############################################################
            

    test_dataset = dataset_cortical_surfaces_segmentation(data_path=data_path,
                                            config=config,
                                            labels_path=labels_path,
                                            split='test',
                                            split_cv=split_cv)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False, 
                                            num_workers=32)
    
    train_dataset.logging()
    
    if split_cv ==-1:

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Validation data: {}'.format(len(val_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, val_loader, test_loader
    
    else:

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, test_loader
    

'''
def loader_numpy(data_path,
                 sampler,
                 bs,
                 bs_val):


    train_data = np.load(os.path.join(data_path,'train_data.npy'))
    train_label = np.load(os.path.join(data_path,'train_labels.npy'))

    print('training data: {}'.format(train_data.shape))

    train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                    torch.from_numpy(train_label).float())


    val_data = np.load(os.path.join(data_path,'validation_data.npy'))
    val_label = np.load(os.path.join(data_path,'validation_labels.npy'))

    print('validation data: {}'.format(val_data.shape))

   
    val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                    torch.from_numpy(val_label).float())


    if sampler:
        print('using sampler...')
        sampler = sampler_preterm(train_label)
        train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                batch_size = bs,
                                                sampler=sampler,
                                                num_workers=16)
    else:
        print('not using sampler...')
        train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                batch_size = bs,
                                                shuffle=True,
                                                num_workers=16)

    val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)
    
    test_data = np.load(os.path.join(data_path,'test_data.npy'))
    test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)

    print('testing data: {}'.format(test_data.shape))

    test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                    torch.from_numpy(test_label).float())

    test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)

    return train_loader, val_loader, test_loader
'''