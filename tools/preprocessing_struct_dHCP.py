# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2022-01-12 15:37:28

'''
This file is used to preprocess data surface metrics into triangular patches
filling the entire surface. 
inputs: (N,C) - N subjects. C channels
outputs: (L,V,C) - L sequence lenght, V number of vertices per patch, C channels
'''

import numpy as np
import pandas as pd
import nibabel as nb

import yaml
import os
import argparse

def main(config):

    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)
    print('')
    print('Using Train/Test/Validation split')
    print('')

    #### PARAMETERS #####

    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    # data path to right and left hemispheres
    path_to_data_left = config['data']['data_path_left']
    path_to_data_right = config['data']['data_path_right']
    
    label_path = config['data']['label_path']
    C = config['data']['channels']
    output_folder = config['output']['folder']

    nbr_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']
    nbr_triangles = config['sub_ico_{}'.format(sub_ico)]['num_patches']

    ####

    train_ids = pd.read_csv(os.path.join(label_path, 'train.csv'))['ids'].to_numpy().reshape(-1)
    train_labels = pd.read_csv(os.path.join(label_path, 'train.csv'))['labels'].to_numpy().reshape(-1)
    nbr_subjects_train = train_ids.shape[0]

    val_ids = pd.read_csv(os.path.join(label_path, 'validation.csv'))['ids'].to_numpy().reshape(-1)
    val_labels = pd.read_csv(os.path.join(label_path, 'validation.csv'))['labels'].to_numpy().reshape(-1)
    nbr_subjects_val = val_ids.shape[0]

    test_ids = pd.read_csv(os.path.join(label_path, 'test.csv'))['ids'].to_numpy().reshape(-1)
    test_labels = pd.read_csv(os.path.join(label_path, 'test.csv'))['labels'].to_numpy().reshape(-1)
    nbr_subjects_test = test_ids.shape[0]
    
    print('')

    data = []

    for i,id in enumerate(train_ids):
        print(id)
        filename = os.path.join(path_to_data_left,'{}_left_merged.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])
        filename = os.path.join(path_to_data_right,'{}_right_merged_ico6_flip.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])
        
    data = np.asarray(data)

    ## data normalisation to 0 mean and variance 1 
    means  = np.mean(np.mean(data,axis=2),axis=0)
    stds  = np.std(np.std(data,axis=2),axis=0)

    normalised_data = (data - means.reshape(1,4,1))/stds.reshape(1,4,1)

    indices_triangles = pd.read_csv('./triangle_indices_ico_{}_sub_ico_{}.csv'.format(ico,sub_ico))

    #shape of the data is nbr_subjects * 20 trianglas * nbr_vertices_per_triangle * channels
    data_train = np.zeros((nbr_subjects_train*2, C, nbr_triangles, nbr_vertices))

    for i,id in enumerate(train_ids):
        print(id)
        for j in range(nbr_triangles):
            indices_to_extract = indices_triangles[str(j)].to_numpy()
            data_train[i,:,j,:] = normalised_data[2*i][:,indices_to_extract]
            data_train[i+nbr_subjects_train,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]
    

    print('')
    print('#'*30)
    print('#Saving: training data')
    print('#'*30)
    print('')
    
    filename = os.path.join(output_folder,'train_data.npy')
    np.save(filename,data_train)
    filename = os.path.join(output_folder,'train_label.npy')
    labels = np.concatenate((train_labels,train_labels))
    np.save(filename,labels)
    
    ### validation

    data = []

    for i,id in enumerate(val_ids):
        print(id)
        filename = os.path.join(path_to_data_left,'{}_left_merged.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])
        filename = os.path.join(path_to_data_right,'{}_right_merged_ico6_flip.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])

    data = np.asarray(data)

    #using the mean and std values from the training set
    normalised_data = (data - means.reshape(1,4,1))/stds.reshape(1,4,1)

    data_val = np.zeros((nbr_subjects_val*2, C, nbr_triangles, nbr_vertices))

    for i,id in enumerate(val_ids):
        print(id)
        for j in range(nbr_triangles):
            indices_to_extract = indices_triangles[str(j)].to_numpy()
            data_val[i,:,j,:] = normalised_data[2*i][:,indices_to_extract]
            data_val[i+nbr_subjects_val,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]

    print('')
    print('#'*30)
    print('#Saving: validation data')
    print('#'*30)
    print('')
    
    filename = os.path.join(output_folder,'validation_data.npy')
    np.save(filename,data_val)
    filename = os.path.join(output_folder,'validation_label.npy')
    np.save(filename,np.concatenate((val_labels,val_labels)))

    ###testing
    data = []

    for i,id in enumerate(test_ids):
        print(id)
        filename = os.path.join(path_to_data_left,'{}_left_merged.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])
        filename = os.path.join(path_to_data_right,'{}_right_merged_ico6_flip.shape.gii'.format(id))
        data.append(np.array(nb.load(filename).agg_data())[:C,:])

    data = np.asarray(data)
    
    #using the mean and std valeus from the training set
    normalised_data = (data - means.reshape(1,4,1))/stds.reshape(1,4,1)

    data_test = np.zeros((nbr_subjects_test*2, C, nbr_triangles, nbr_vertices))

    for i,id in enumerate(test_ids):
        print(id)
        for j in range(nbr_triangles):
            indices_to_extract = indices_triangles[str(j)].to_numpy()
            data_test[i,:,j,:] = normalised_data[2*i][:,indices_to_extract]
            data_test[i+nbr_subjects_test,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]

    print('')
    print('#'*30)
    print('#Saving: test data')
    print('#'*30)
    print('')


    filename = os.path.join(output_folder,'test_data.npy')
    np.save(filename,data_test)
    filename = os.path.join(output_folder,'test_label.npy')
    np.save(filename,np.concatenate((test_labels,test_labels)))
    
    print(data_train.shape)
    print(data_val.shape)
    print(data_test.shape)


if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='preprocessing HCP data for patching')
    
    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config)