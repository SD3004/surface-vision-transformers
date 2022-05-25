# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-03-05 17:25:44
import os 
import argparse
import yaml
import sys

sys.path.append('../')
sys.path.append('./')

import torch
import numpy as np
import pandas as pd

from models.sit import SiT


def test(config):

    gpu = config['testing']['gpu']
    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']
    data_path = config['data']['data_path'].format(ico,sub_ico)
    bs_test = config['testing']['bs_test']
    folder_to_ckpt = config['testing']['folder']

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']

    test_data = np.load(os.path.join(data_path,'test_data.npy'))
    test_label = np.load(os.path.join(data_path,'test_labels.npy'))

    print('testing data: {}'.format(test_data.shape))

    test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                    torch.from_numpy(test_label).float())

    test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                            batch_size = bs_test,
                                            shuffle=False,
                                            num_workers=8)


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


    print('loading model')
    model.load_state_dict(torch.load(os.path.join(folder_to_ckpt,'checkpoint.pth'),map_location=device))

    model.to(device)

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

        preds_ = np.array(preds_).reshape(-1)
        targets_ = np.array(targets_).reshape(-1)

        df = pd.DataFrame()
        df['pred']=preds_
        df['targets']=targets_
        print('Saving results...')
        df.to_csv(os.path.join(folder_to_ckpt,'results.csv'), index=False)

        print('| TESTING RESULTS | MAE - {} |'.format(mae_test_epoch))

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
    test(config)
