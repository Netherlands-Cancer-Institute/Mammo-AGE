import os
import math
import torch
import shutil
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(results_dir, state, is_best):
    filename = results_dir + '/model_last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, results_dir + '/model_best.pth.tar')


# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr

    if args.cos:  # cosine lr schedule
        lr *= 0.1 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """ 
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """


def seed_reproducer(seed=5):
    import random
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True