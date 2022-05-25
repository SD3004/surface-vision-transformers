# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2021-12-16 10:41:47
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file contains the implementation of the ViT model: https://arxiv.org/abs/2010.11929 adapted to the case of surface patching. 
Input data is a sequence of non-overlapping patches. 
'''

import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer, Attention, FeedForward, PreNorm

class SiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        mlp_dim,
                        pool = 'cls', 
                        num_patches = 20,
                        num_classes= 1,
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_dim = num_channels * num_vertices

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)