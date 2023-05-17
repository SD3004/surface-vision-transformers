# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-02-22 11:35:57
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file contains our implementation of the ViT model: https://arxiv.org/abs/2010.11929
'''

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer, Attention, FeedForward, PreNorm

from timm.models.layers import trunc_normal_

class SiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        pool = 'cls', 
                        num_patches = 20,
                        num_classes= 1,
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.,
                        bottleneck_dropout = 0.,
                        mlp_ratio = 4,
                        use_pe = True,
                        use_confounds = False,
                        use_bottleneck= False, 
                        weights_init = False, 
                        use_class_token = True,
                        trainable_pos_emb = True, 
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_dim = num_channels * num_vertices

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        if use_bottleneck:
            self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Dropout(bottleneck_dropout),
            nn.Linear(patch_dim,1024),
            nn.Dropout(bottleneck_dropout),
            nn.Linear(1024, dim),
        )

        if use_bottleneck:
            print('using bottleneck')

        if use_confounds:
            self.proj_confound = nn.Sequential(
                                    nn.BatchNorm1d(1),
                                    nn.Linear(1,dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.use_pe = use_pe
        self.use_confounds = use_confounds

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_ratio*dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        if weights_init: # apply the same weight init as timm's github
            print('Using initialisation from Timms repo')
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_class_token else None
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim, requires_grad=trainable_pos_emb) * .02) if use_pe else None
            self.init_weights()

        else: # apply to weight init from vit_pytorch
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if use_class_token else None
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim), requires_grad=trainable_pos_emb) if use_pe else None

    def init_weights(self,):

        trunc_normal_(self.pos_embedding, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6) 

        # initialize nn.Linear
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)


    def forward(self, img, confounds=None):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        if self.use_confounds and (confounds is not None):
            confounds = self.proj_confound(confounds.view(-1,1))
            confounds = repeat(confounds, 'b d -> b n d', n=n+1)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        x = torch.cat((cls_tokens, x), dim=1)
        if self.use_pe: 
            x += self.pos_embedding[:, :(n + 1)]
        if self.use_confounds and (confounds is not None):
            x += confounds
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)

if __name__ == '__main__':

    model = SiT(dim = 192,
                depth = 12,
                heads= 3,
                weights_init=True)

