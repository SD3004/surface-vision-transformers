import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
import nibabel as nb
import pandas as pd
from timm.models.layers import trunc_normal_
from vit_pytorch.vit import Transformer

torch.autograd.set_detect_anomaly(True)

'''
Code adapted to surfaces from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
'''

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        use_pos_embedding_decoder = True,
        loss = 'mse',
        use_all_patch_loss = False,
        mask = True, 
        dataset = 'dHCP',
        configuration = 'template',
        sampling = 'msm',
        sub_ico = 3, 
        num_channels =1, 
        weights_init = False,
        path_to_template = '',
        path_to_workdir = ''

    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.num_patches, encoder.encoding_dim
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1] # channels*dim

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim) #fixed embedding
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.use_pos_emb_decoder = use_pos_embedding_decoder
        self.loss = loss
        self.use_all_patch_loss = use_all_patch_loss
        self.masking = mask
        self.dataset = dataset
        self.num_channels = num_channels
        self.configuration = configuration
        self.num_vertices_per_channel = pixel_values_per_patch // self.num_channels
        self.decoder_dim = decoder_dim

        if self.use_all_patch_loss:
            print('Using loss on all patches')
        else:
            print('Using loss only on masked patches')

        if self.encoder.use_pe:
            print('Using positional embedding in encoder MAE')
        else:
            print('Not using positional embedding in encoder MAE')

        if self.use_pos_emb_decoder:
            print('Using positional embedding in decoder MAE')
        else:
            print('Not using positional embedding in decoder MAE')

        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template': # for dHCP
            self.mask =np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(path_to_template)).agg_data())
        elif self.masking and self.dataset == 'UKB': # for UKB
            self.mask = np.array(nb.load('{}/L.atlasroi.ico6_fs_LR.shape.gii'.format(path_to_template)).agg_data())
        elif self.masking and self.dataset == 'HCP': # for UKB
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(path_to_template)).agg_data())

        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,sampling,sub_ico))

        if weights_init:
            self.mask_token = nn.Parameter(torch.zeros(decoder_dim))
            self.init_weights()
        else:
            self.mask_token = nn.Parameter(torch.randn(decoder_dim))

    def init_weights(self,):
        nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.encoder.use_pe:
            tokens = tokens + self.encoder.pos_embedding[:, self.encoder.num_prefix_tokens:(num_patches + self.encoder.num_prefix_tokens)] #can be set to fixed in the encoder #no class token
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        num_unmasked = num_patches - num_masked
        rand_indices = torch.rand(batch, num_patches, device = device, requires_grad=False).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        unmasked_patches = patches[batch_range, unmasked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        if self.use_pos_emb_decoder:
            unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        else:
            unmasked_decoder_tokens = decoder_tokens

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        if self.use_pos_emb_decoder:
            mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        ###v0 - concatenate the mask with unmasked
        #decoder_tokens = torch.cat((mask_tokens, unmasked_decoder_tokens), dim = 1)
        ###v1 - reorder the sequence
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        ##v0
        #mask_tokens = decoded_tokens[:, :num_masked]
        #pred_pixel_values = self.to_pixels(mask_tokens)
        #unmask_tokens = decoded_tokens[:, num_masked:]
        #pred_pixel_values_unmasked = self.to_pixels(unmask_tokens)
        ##v1
        mask_tokens = decoded_tokens[batch_range,masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        unmask_tokens = decoded_tokens[batch_range,unmasked_indices]
        pred_pixel_values_unmasked = self.to_pixels(unmask_tokens)

        # calculate reconstruction loss

        prediction_pixels_mask = pred_pixel_values.detach()
        prediction_pixels_unmasked  = pred_pixel_values_unmasked.detach()

        if self.loss == 'mse':
            if self.use_all_patch_loss:
                recon_loss = F.mse_loss(torch.cat((pred_pixel_values, pred_pixel_values_unmasked), dim=1), torch.cat((masked_patches,unmasked_patches), dim=1)) 
            else:    
                recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        elif self.loss =='l1':
            if self.use_all_patch_loss:
                recon_loss = F.l1_loss(torch.cat((pred_pixel_values, pred_pixel_values_unmasked), dim=1), torch.cat((masked_patches,unmasked_patches), dim=1)) 
            else:    
                recon_loss = F.l1_loss(pred_pixel_values, masked_patches)
        elif self.loss == 'smoothl1':
            if self.use_all_patch_loss:
                recon_loss = F.smooth_l1_loss(torch.cat((pred_pixel_values, pred_pixel_values_unmasked), dim=1), torch.cat((masked_patches,unmasked_patches), dim=1)) 
            else:
                recon_loss = F.smooth_l1_loss(pred_pixel_values, masked_patches, beta=0.05)

        #### ADDED #### Mask the patches with the cut
        mask = torch.Tensor(self.mask)
        mask = mask.to(device)

        prediction_pixels_mask = rearrange(prediction_pixels_mask, 'b n (v c) -> b n c v', b =batch, n=num_masked, c =self.num_channels,v=self.num_vertices_per_channel)

        prediction_pixels_mask_ = torch.zeros_like(prediction_pixels_mask)

        prediction_pixels_unmasked = rearrange(prediction_pixels_unmasked, 'b n (v c) -> b n c v', b =batch, n=num_unmasked, c =self.num_channels,v=self.num_vertices_per_channel)

        prediction_pixels_unmasked_ = torch.zeros_like(prediction_pixels_unmasked)

        for i in range(batch):
            for j,k in enumerate(masked_indices[i]):
                indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                indices_to_extract = indices_to_extract.to(device)
                prediction_pixels_mask_[i,j,:,:] = prediction_pixels_mask[i,j,:,:] * mask[indices_to_extract]
        prediction_pixels_mask_ = rearrange(prediction_pixels_mask_, 'b n c v -> b n (v c)')


        for i in range(batch):
            for j,k in enumerate(unmasked_indices[i]):
                indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                indices_to_extract = indices_to_extract.to(device)
                prediction_pixels_unmasked_[i,j,:,:] = prediction_pixels_unmasked[i,j,:,:] * mask[indices_to_extract]
        prediction_pixels_unmasked_ = rearrange(prediction_pixels_unmasked, 'b n c v -> b n (v c)')


        return recon_loss, prediction_pixels_mask_, prediction_pixels_unmasked_, masked_indices, unmasked_indices