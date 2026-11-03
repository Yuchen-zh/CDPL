"""CDPL: multi-view long-tailed semi-supervised classification.

Entry point. Dataset paths are resolved relative to the repository root, so the
project can be cloned anywhere.
"""

import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
from ruamel.yaml import YAML

from trainers.train1 import train

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(args):
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    start = time.time()
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.data_base is None:
        args.data_base = args.data_base_local
    if not os.path.isabs(args.data_base):
        args.data_base = os.path.join(REPO_ROOT, args.data_base)

    print(f'LR: {args.lr}, Decay: {args.weight_decay}')
    print('=' * 81)
    print('Start Train')
    print('=' * 81)
    torch.cuda.empty_cache()

    train(args)

    elapsed = time.time() - start
    print('Total running time: {:.0f}h {:.0f}m'.format(elapsed // 3600, (elapsed % 3600) // 60))


def over_write_args_from_file(args, yml):
    yaml = YAML(typ='rt')
    with open(yml, 'r', encoding='utf-8') as handle:
        for key, value in yaml.load(handle).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)


def get_config():
    parser = argparse.ArgumentParser(description="Multi-View Long-tail Semi-Supervised Learning")

    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda device, e.g. cuda:0")
    parser.add_argument('--train', type=str, choices=["True", "False"], default="False")
    parser.add_argument('--test', type=str, choices=["True", "False"], default="False")
    parser.add_argument('--config_file', type=str, required=True,
                        help="config file name under configs/, e.g. CIFAR100/0.09_exp_0.1_CIFAR100.yaml")
    parser.add_argument('--ratio', type=float, required=True, help="ratio of labeled data")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data_base', type=str, default=None,
                        help="dataset root; overrides data_base_local from the config")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="checkpoint directory; defaults to checkpoint_{ratio}_{seed}")

    args = parser.parse_args()
    args.train = args.train == "True"
    args.test = args.test == "True"
    args.ulb_ratio = args.ratio

    config_path = os.path.join(REPO_ROOT, 'configs', args.config_file)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found!")
    over_write_args_from_file(args, config_path)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(REPO_ROOT, f'checkpoint_{args.ratio}_{args.seed}')

    return args


if __name__ == "__main__":
    args = get_config()
    run(args)
