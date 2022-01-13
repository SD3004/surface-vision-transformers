#
# Created on Wed Oct 20 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import nibabel as nb

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