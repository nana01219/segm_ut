import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import copy

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter, create_segmenter_uncertainty, create_segmenter_ut_10, create_segmenter_each
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate

def load_part(model_dict, checkpoint, part):
    if part == "backbone":
        for k, v in checkpoint.items():
            print("the k", k)
            flag = False
            for ss in  model_dict.keys():            
                if k == ss:
                    print("MATCH:", k)
                    flag = True
                    break
                else:
                    continue
            if flag:
                model_dict[ss] = checkpoint[k]
        return model_dict
    elif part == "uncertainty":
        for k, v in checkpoint.items():
            flag = False
            for ss in  model_dict.keys():
                if k in ss:
                    if "block_data" in k:
                        flag = True
                        break  
                    else:
                        break               
            if flag:
                print("Reserve:", ss, k)
            else:
                model_dict[ss] = checkpoint[k]
        return model_dict
    else:
        raise Exception("Uncertainty: Model do not have such parts")



                

def main(
    local_rank,
    log_dir,
    dataset,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    ut,
    ug,
    ft,
    use_norm,
    pre_ck,
    pre_epoch,
    amp,
    resume,
):
    # start distributed mode
    ptu.local_rank = local_rank
    ptu.dist_rank = local_rank
    ptu.set_gpu_mode(True)
    distributed.init_process()

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth"

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    
    if ut == 0:
        model = create_segmenter(net_kwargs, with_ut = False)
    elif ut == -1:
        model = create_segmenter_each(net_kwargs)
        # model = create_segmenter(net_kwargs, with_ut = True)
    elif ut == 1:
        model = create_segmenter_uncertainty(net_kwargs, block_type="block_data")
    elif ut == 11:
        model = create_segmenter_uncertainty(net_kwargs, block_type="block_dropout")
    elif ut <= 10 and ut > 1:
        model = create_segmenter_ut_10(net_kwargs, repeat_num=ut)

    for k in model.state_dict():
        print(k)
    # input()
    model.to(ptu.device)


    # load the pre-trained MASKtransformer and fix the parameters
    if pre_ck is not None:  
        assert (pre_epoch > 0)  & (pre_epoch <= ft)
        model_dict = model.state_dict()
        random_dict = copy.deepcopy(model.state_dict())
        checkpoint_mask = torch.load(pre_ck, map_location=ptu.device)['model']

        model_dict = load_part(model_dict, checkpoint_mask, "backbone")
        model.load_state_dict(model_dict)
        for name, param in model.named_parameters():
            if "block_data" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        

    # optimizer
    
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    if pre_ck is not None: 
        optimizer_kwargs["iter_max"] = len(train_loader) * pre_epoch
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)

    lr_scheduler = create_scheduler(opt_args, optimizer)
    if pre_ck is not None: 
        opt_vars["iter_max"] = len(train_loader) * (optimizer_kwargs["epochs"] - pre_epoch)
    lr_scheduler_2 = create_scheduler(opt_args, optimizer)

    
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
    else:
        sync_model(log_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    use_gate = False
 
    for epoch in range(start_epoch, num_epochs):
        """if ft == 2:
            if epoch > 24:
                for name, param in model.named_parameters():
                    if "uncertainty" in name:
                        param.requires_grad = False
                        print("******Uncertainty layer now requires_grad = False")
        elif ft == 3:
            if epoch % 2 == 0:
                for name, param in model.named_parameters():
                    if "uncertainty" in name:
                        param.requires_grad = True
                        print("******Uncertainty layer now requires_grad = True")
            else:
                for name, param in model.named_parameters():
                    if "uncertainty" in name:
                        param.requires_grad = False
                        print("******Uncertainty layer now requires_grad = False")"""
        if ug == -1:
            print("***Warning: Always use gate")
            use_gate = True 
        elif ug == 0:
            use_gate = False
        elif ug > 0:
            if epoch > ug:
                use_gate = True 
                print("******* Now notice that use_gate=True")

        if ft > 0:
            if epoch > ft:
                for name, param in model.named_parameters():
                    if "uncertainty" in name:
                        param.requires_grad = False
                        print("******Uncertainty layer now requires_grad = False")
                    else:
                        param.requires_grad = True

        if pre_epoch > 0:
            if epoch == pre_epoch:
                lr_scheduler = lr_scheduler_2
                model_dict = model.state_dict()
                model_dict = load_part(model_dict, random_dict, "uncertainty")
                # for k, v in model_dict.items():
                #     print("dict:", k)
                # for name, param in model.named_parameters():
                #     print("model-", name)
                model.load_state_dict(model_dict)

                # if ptu.distributed:
                #     model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

                for name, param in model.named_parameters():
                    param.requires_grad = True
                
                # print("***")

        # if use_norm:
        #     norm_flag = not use_gate

        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            use_gate = use_gate,
            use_norm = use_norm,
        )

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            torch.save(snapshot, checkpoint_path)
            torch.cuda.empty_cache()

        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
                use_gate = use_gate,
            )
            print(f"Stats [{epoch}]:", eval_logger, flush=True)
            print("")
            torch.cuda.empty_cache()

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)

@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--ut", default=None, type=int, help="-1 each 0 no uncertainty 1 uncertainty >1 repeat number")
@click.option("--ug", default=0, type=int, help="if >0, from epoch ug the model will use 0-1 gate; but ug = 0 always don't use")
@click.option("--ft", default=-1, type=int, help="if >0, from epoch ft the model will fix uncertainty module")
@click.option("--use_norm", default="-1/0.015", type=str)
@click.option("--pre_ck", default=None, type=str)
@click.option("--pre_epoch", default=0, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)

def muti(
    log_dir,
    dataset,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    ut,
    ug,
    ft,
    use_norm,
    pre_ck,
    pre_epoch,
    amp,
    resume,
):
    mp.spawn(main,
        args=(log_dir,
            dataset,
            im_size,
            crop_size,
            window_size,
            window_stride,
            backbone,
            decoder,
            optimizer,
            scheduler,
            weight_decay,
            dropout,
            drop_path,
            batch_size,
            epochs,
            learning_rate,
            normalization,
            eval_freq,
            ut,
            ug,
            ft,
            use_norm,
            pre_ck,
            pre_epoch,
            amp,
            resume,
        ),
        nprocs=ptu.world_size,
        join=True)


if __name__ == "__main__":  
    muti()
