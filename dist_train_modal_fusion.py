# System libs
import datetime

import os
import os.path as osp

import random
import pickle
import argparse

# Numerical libs
import torch
import torch.nn as nn

import torch.utils.data as data_utils
import pandas as pd
import numpy as np

# Our libs
from configs.defaults import _C as train_config
from utils import setup_logger

import torch.distributed as dist
from models.mil_net import MILFusion
from dataloader.feat_bag_dataset import ModalFusionDataset

from metrics import ROC_AUC


import matplotlib
from typing import Tuple

old_print = print
from rich import print


matplotlib.use("Agg")

def print_in_main_thread(msg: str,):
    if local_rank == 0:
        print(msg)

def log_in_main_thread(msg: str):
    if local_rank == 0:
        logger.info(msg)


def evaluate(model: nn.Module, val_loader, epoch, local_rank, final_test=False, dump_dir=None) -> Tuple[float, float, float]:
    """
    distributed method for model inference, meter will automatically deal with the sync of multiple gpus
    Parameters
    ----------
    model
    val_loader
    epoch
    local_rank

    Returns
    -------

    """

    auc_meter = ROC_AUC()
    model.eval()

    start_time = datetime.datetime.now()
    print(f'Start test at {start_time} at {local_rank}')

    with torch.no_grad():
        for batch_nb, batch_data in enumerate(val_loader):
            if local_rank == 0:
                old_print(f'\r {batch_nb} / {len(val_loader)} ', end='')

            label = batch_data['label'].cuda(local_rank)
            label = label.view(label.size(0), 1).float()
            output, loss, __ = model(batch_data)

            auc_meter.update([torch.sigmoid(output.detach()).cpu().view(-1), label.view(-1).detach().cpu()])

    print()

    end_time = datetime.datetime.now()
    print(f'End test at {end_time} at {local_rank}')
    all_pred = auc_meter.predictions
    print(all_pred)



def main(cfg, local_rank):
    """
    build
    prepare model training
    :param cfg:
    :param local_rank:
    :return:
    """
    if local_rank == 0:
        logger.info(f'Build model')

    with open(cfg.dataset.tab_data_path, 'rb') as infile:
        tab_data = pickle.load(infile)
    cat_dims = tab_data['cat_dims']
    cat_idxs = tab_data['cat_idxs']

    tab_data_df = pd.read_csv(cfg.dataset.tab_data_path.rsplit('.', 1)[0] + '.csv')


    df_path = cfg.dataset.df_path
    if local_rank == 0:
        print(f'Load df from {os.path.abspath(df_path)}')
    df = pd.read_csv(df_path)
    test_df = df[df.split == 'test']

    test_data_df = tab_data_df[tab_data_df.split == 'test']

    if local_rank == 0:
        logger.info(f'Build dataset')

    """build dataset"""

    test_dataset = ModalFusionDataset(
        cli_feat=test_df,
        cli_data=test_data_df,
        scale1_feat_root=cfg.dataset.scale1_feat_root,
        scale2_feat_root=cfg.dataset.scale2_feat_root,
        scale3_feat_root=cfg.dataset.scale3_feat_root,
        select_scale=cfg.dataset.select_scale,
        cfg=cfg,
        shuffle_bag=False,
        is_train=False
    )

    log_in_main_thread('Dataset load finish')
    test_sampler = data_utils.distributed.DistributedSampler(test_dataset, rank=local_rank)

    num_workers = cfg.train.workers

    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        sampler=test_sampler
    )

    """build model"""
    log_in_main_thread('Build model')
    if hasattr(cfg.model, 'fusion_method'):
        fusion = cfg.model.fusion_method
    else:
        fusion = 'mmtm'
    if hasattr(cfg.model, 'use_k_agg'):
        use_k_agg = cfg.model.use_k_agg
        k_agg = cfg.model.k_agg
    else:
        use_k_agg = False
        k_agg = 10

    if cfg.model.arch == 'm3d':
        logger.info(f'Adapt m3d')
        from models.mil_net import M3D
        model = M3D(img_feat_input_dim=1280,
                          tab_feat_input_dim=32,
                          img_feat_rep_layers=4,
                          num_modal=cfg.model.num_modal,
                          fusion=fusion,
                          use_tabnet=cfg.model.use_tabnet,
                          use_k_agg=use_k_agg,
                          k_agg=k_agg,
                          tab_indim=test_dataset.tab_data_shape,
                          cat_dims=cat_dims,
                          cat_idxs=cat_idxs,
                          local_rank=local_rank)
    elif cfg.model.arch == 'attention_refine':
        logger.info(f'attention_refine')
        from models.mil_net import MILFusionAppend
        model = MILFusionAppend(img_feat_input_dim=1280,
                          tab_feat_input_dim=32,
                          img_feat_rep_layers=4,
                          num_modal=cfg.model.num_modal,
                          fusion=fusion,
                          use_tabnet=cfg.model.use_tabnet,
                          use_k_agg=use_k_agg,
                          k_agg=k_agg,
                          tab_indim=test_dataset.tab_data_shape,
                          cat_dims=cat_dims,
                          cat_idxs=cat_idxs,
                          local_rank=local_rank)
    elif cfg.model.arch == 'attention_add':
        logger.info(f'attention_add')
        from models.mil_net import MILFusionAdd
        model = MILFusionAdd(img_feat_input_dim=1280,
                          tab_feat_input_dim=32,
                          img_feat_rep_layers=4,
                          num_modal=cfg.model.num_modal,
                          fusion=fusion,
                          use_tabnet=cfg.model.use_tabnet,
                          use_k_agg=use_k_agg,
                          k_agg=k_agg,
                          tab_indim=test_dataset.tab_data_shape,
                          cat_dims=cat_dims,
                          cat_idxs=cat_idxs,
                          local_rank=local_rank)
    else:
        model = MILFusion(img_feat_input_dim=1280,
                          tab_feat_input_dim=32,
                          img_feat_rep_layers=4,
                          num_modal=cfg.model.num_modal,
                          fusion=fusion,
                          use_tabnet=cfg.model.use_tabnet,
                          use_k_agg=use_k_agg,
                          k_agg=k_agg,
                          tab_indim=test_dataset.tab_data_shape,
                          cat_dims=cat_dims,
                          cat_idxs=cat_idxs,
                          local_rank=local_rank)

    
    model = model.cuda(local_rank)
    model = model.to(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    if local_rank == 0:
        logger.info(f'Start training')

    """
    load best ckpt
    """
    if hasattr(cfg.test, 'checkpoint'):
        ckpt_path = cfg.test.checkpoint
        if osp.exists(ckpt_path):
            bst_val_model_path = ckpt_path

    log_in_main_thread(f'Load model from {bst_val_model_path}')

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(bst_val_model_path, map_location=map_location))

    model.eval()

    evaluate(model, test_loader, cfg.train.num_epoch, local_rank, final_test=True,
                                             dump_dir=cfg.save_dir)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch WSI Multi modal training"
    )
    parser.add_argument(
        "--cfg",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    cfg = train_config
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    local_rank = args.local_rank
    # set dist
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank)
    
    print(f'local rank: {args.local_rank}')

    time_now = datetime.datetime.now()
    cfg.save_dir = osp.join(cfg.save_dir,
                            f'{time_now.year}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}')


    if not os.path.isdir(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)
    logger = setup_logger(distributed_rank=args.local_rank, filename=osp.join(cfg.save_dir, 'train_log.txt'))  # TODO
    log_in_main_thread(f'Save result to : {cfg.save_dir}')


    if args.local_rank == 0:
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))
        logger.info("Outputing checkpoints to: {}".format(cfg.save_dir))
        with open(os.path.join(cfg.save_dir, 'config.yaml'), 'w') as f:
            f.write("{}".format(cfg))

    num_gpus = 1

    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    main(cfg, args.local_rank)


