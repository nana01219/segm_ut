import os
import torch


"""
GPU wrappers

CUDA_VISIBLE_DEVICES=3 python -m segm.train --log-dir test --dataset ade20k   --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 1
CUDA_VISIBLE_DEVICES=3 python -m segm.train --log-dir test --dataset ade20k   --backbone vit_base_patch16_384 --decoder mask_transformer --ut -0 --ft 1 --ck ./B_16.pth --sp_one no_relation
CUDA_VISIBLE_DEVICES=2,3 python -m segm.train_muti --log-dir tiny_ug0_ft24_pe24_e88_muti  --dataset ade20k   \
    --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 1  --ft 24 --pre_ck ./Tiny_16.pth --pre_epoch 24

CUDA_VISIBLE_DEVICES=0,1 python -m segm.train_muti --log-dir tiny_ug24_ft24_pe24_e88_muti  --dataset ade20k   \
    --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 1 --ug 24 --ft 24 --pre_ck ./Tiny_16.pth --pre_epoch 24
"""

use_gpu = False
local_rank = 0
device = None

distributed = False
dist_rank = 0
world_size = 4


def set_gpu_mode(mode):
    global use_gpu
    global device
    global local_rank
    global distributed
    global dist_rank
    global world_size
    # local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    # dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    # world_size = int(os.environ.get("SLURM_NTASKS", 1))

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
