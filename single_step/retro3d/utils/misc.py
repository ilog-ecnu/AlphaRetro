import torch
from collections import OrderedDict
import time
import os
from contextlib import redirect_stdout
import pathlib
import logging
import numpy as np
import random
import torch.backends.cudnn as cudnn

from models.model import Retro3D

def get_saved_info(path):
    check_point = torch.load(path, map_location=torch.device('cpu'))
    return check_point['settings'], check_point['model']

def get_pretrained_model(config, model_dict, data):
    model = Retro3D(
        n_src_vocab=len(data.src_t2i),
        n_trg_vocab=len(data.tgt_t2i),
        src_pad_idx=data.src_t2i['<pad>'],
        tgt_pad_idx=data.tgt_t2i['<pad>'],
        d_model=config.MODEL.D_MODEL,
        d_inner=config.MODEL.D_INNER,
        n_enc_layers=config.MODEL.N_LAYERS,
        n_dec_layers=config.MODEL.N_LAYERS,
        n_head=config.MODEL.N_HEAD,
        dropout=config.MODEL.DROPOUT,
        shared_embed=config.MODEL.SHARED_EMBED,
        shared_encoder=config.MODEL.SHARED_ENCODER
    )

    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model
