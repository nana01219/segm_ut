import os
import torch


"""
GPU wrappers
CUDA_VISIBLE_DEVICES=0 python -m segm.train --log-dir uncertainty_tiny_mask --dataset ade20k   --backbone uncertainty_vit_tiny_patch16_384 --decoder mask_transformer --ut 1
CUDA_VISIBLE_DEVICES=1 python -m segm.train --log-dir compare_tiny_vit --dataset ade20k   --backbone compare_vit_tiny_patch16_384 --decoder mask_transformer --ut 0
CUDA_VISIBLE_DEVICES=1 python -m segm.train --log-dir abcde --dataset ade20k   --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 0
CUDA_VISIBLE_DEVICES=0,1 python -m segm.train --log-dir pre-tiny_vit_8 --dataset ade20k   --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 0
CUDA_VISIBLE_DEVICES=3 python -m segm.train --log-dir test --dataset ade20k   --backbone vit_tiny_patch16_384 --decoder mask_transformer --ut 1
CUDA_VISIBLE_DEVICES=2 python -m segm.train --log-dir test --dataset ade20k   --backbone vit_base_patch16_384 --decoder mask_transformer --ut 1 --ft 1 --ck ./B_16.pth
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
