import os
import shutil
import argparse
import json
import math
import torch
import logging
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Append relative paths to sys.path
sys.path.append(os.path.join(current_dir))
sys.path.append(os.path.dirname(current_dir)) # add father path ../
sys.path.append(os.path.join(current_dir, 'losses'))
sys.path.append(os.path.join(current_dir, 'learning'))
sys.path.append(os.path.join(current_dir, 'models'))
# sys.path.append(os.path.join(current_dir, 'utils'))
from utils.utils import *
from utils.mylogging import open_log
from learning.train_val_test_demo import get_predict_demo
from dataload.combine_data_loader import dataloador
from dataload.custom_dataset_4views import CustomDataset
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser(description='Mammo-AGE-Test')

    # Model parameters
    # ---------------------------------
    parser.add_argument('-a', '--arch', default='resnet18', help='[resnet18, resnet50, efficientnet_b0, densenet121, convnext_tiny]')
    parser.add_argument('--model-method', default='Mammo-AGE', type=str, help='Mammo-AGE model')
    parser.add_argument('--min_age', type=int, default=0, help='min age of the dataset')
    parser.add_argument('--max_age', type=int, default=99, help='max age of the dataset')
    parser.add_argument('--num-output-neurons', type=int, default=100, help='number of ouput neurons of your model, 18-90.')
    parser.add_argument('--nblock', default=10, type=int, help='number of blocks in transformer')
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size in transformer')
    parser.add_argument('--multi_view', action='store_true', default=True, help='If true, will use multi-view model.')
    parser.add_argument('--test_mode', default='predict', type=str, help='test mode in [predict, vis].')

    # Training parameters
    # ---------------------------------
    parser.add_argument('--seed', default=5, type=int, help='seed for initializing training.')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num-workers', default='12', type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--fold', default=0, type=int, help='5-fold cross validation')

    # Dataset
    # ---------------------------------
    parser.add_argument('--dataset', default='VinDr', type=str, help='dataset in [RSNA, VinDr, DDSM, EMBED].')
    parser.add_argument('--img-size', default=1024, type=int, help='size of image')
    parser.add_argument('--dataset-config', default='/Path/to/dataset/config.yaml', type=str, metavar='PATH',
                        help='path to dataset config')
    parser.add_argument('--csv-dir', default='/Path/to/data/base_csv_folder', type=str, metavar='PATH',
                        help='path to base folder for all datasets csv files')
    parser.add_argument('--image-dir', default='/Path/to/data/base_image_folder', type=str, metavar='PATH',
                        help='path to base folder for all datasets image files')

    # Checkpoint
    # ---------------------------------
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH',
                        help='path to cache folder for checkpoint')

    # Loss function parameters for Mammo-AGE model
    # ---------------------------------
    parser.add_argument('--loss-method', default='ce', type=str, help='loss type in [l1, mse, ce].')
    parser.add_argument('--main-loss-type', default='cls', type=str, help='loss type in [cls, reg, rank].')
    parser.add_argument('--lambda_density', default=0.2, type=float, metavar='M',
                        help='lambda of density classification loss (default: 0.2)')

    # For POE method
    # ---------------------------------
    parser.add_argument('--max-t', type=int, default=50, help='number of samples during sto.')
    parser.add_argument('--no-sto', action='store_true', default=False,
                        help='not using stochastic sampling when training or testing.')
    parser.add_argument('--distance', type=str, default='JDistance',
                        help='distance metric between two gaussian distribution')
    parser.add_argument('--alpha-coeff', type=float, default=1e-5, metavar='M', help='alpha_coeff (default: 0)')
    parser.add_argument('--beta-coeff', type=float, default=1e-4, metavar='M', help='beta_coeff (default: 1.0)')
    parser.add_argument('--margin', type=float, default=5, metavar='M', help='margin (default: 1.0)')
    # ---------------------------------


    args = parser.parse_args()

    # Load the arguments from the checkpoint json file
    with open(args.results_dir + '/args.json', 'r') as f:
        raw_dict = json.load(f)


    args.arch = raw_dict['arch']
    args.img_size = raw_dict['img_size']
    args.main_loss_type = raw_dict['main_loss_type']
    args.loss_method = raw_dict['loss_method']
    args.model_method = raw_dict['model_method']
    args.num_output_neurons = raw_dict['num_output_neurons']

    args.fold = raw_dict['fold']
    args.nblock = raw_dict['nblock']
    args.hidden_size = raw_dict['hidden_size']
    args.seed = raw_dict['seed']
    args.use_sto = raw_dict['use_sto']
    args.first_stage_model = raw_dict['first_stage_model']
    args.second_stage = raw_dict['second_stage']

    if args.no_sto:
        args.use_sto = False

    return args


def get_model(args):
    # Import necessary modules
    from models.mammo_age_model import Mammo_AGE
    from losses.mean_variance_loss import MeanVarianceLoss
    from losses.probordiloss import ProbOrdiLoss

    # Check if the model method is Mammo-AGE
    if args.model_method in "Mammo-AGE":
        # Initialize the Mammo-AGE model with specified parameters
        model = Mammo_AGE(arch = args.arch, num_output=args.num_output_neurons, nblock=args.nblock,
                          hidden_size=args.hidden_size, second_stage=args.second_stage,
                          first_stage_model=args.first_stage_model).cuda()

        # Define the first loss criterion
        criterion1 = ProbOrdiLoss(distance=args.distance, alpha_coeff=args.alpha_coeff,
                                  beta_coeff=args.beta_coeff, margin=args.margin,
                                  main_loss_type=args.main_loss_type).cuda()

        # Define the second loss criterion
        criterion2 = MeanVarianceLoss(lambda_1=0.2, lambda_2=0.05, cumpet_ce_loss=False,
                                      start_age=args.min_age).cuda()
        # Combine both criteria into a list
        criterion = [criterion1, criterion2]
    else:
        raise ValueError(f" Model: {args.model_method} is not supported.")

    return model, criterion


def get_data_info(args):
    # Define subsets for your dataset
    subsets = ['train', 'valid', 'test'] # e.g. 'train', 'valid', 'test'
    # Get the CSV directory from arguments
    test_data_info_lists = []
    for subset in subsets:
        if subset == 'train' or subset == 'valid':
            test_data_info_lists.append(f'{args.csv_dir}/{args.dataset}/{subset}_data_info_{args.fold}_fold.csv')
        else:
            test_data_info_lists.append(f'{args.csv_dir}/{args.dataset}/{subset}_data_info.csv')

    return subsets, test_data_info_lists


def main():
    # Parse command-line arguments
    # --------------------------------------
    args = arg_parse()
    seed_reproducer(seed=args.seed)

    # Set up the logger
    # --------------------------------------
    open_log(args, '{}_{}'.format(args.test_mode, args.dataset))
    logging.info(str(args).replace(',', "\n"))

    # define train val test demo
    # ----------------------------------------
    predict = get_predict_demo(args.model_method, vis=False if args.test_mode != 'vis'  else True)  # if vis vis=True
    # ---------------------------------

    # Define Model and loss functions
    # ----------------------------------------
    model, criterion = get_model(args)
    logging.info(model)

    # Load the model from the checkpoint
    # load pretrained model and test
    # ----------------------------------------
    checkpoint = torch.load(args.results_dir + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # dataset prepare
    # ---------------------------------
    subsets, test_data_info_lists = get_data_info(args)
    loaders = {}

    for i in range(len(subsets)):
        subset = subsets[i]
        loaders[subset] = dataloador(CustomDataset, [test_data_info_lists[subset]], [args.dataset], args)
    logging.info('finish data loader')
    # --------------------------------------

    for i, name_loader in enumerate(loaders):
        loader = loaders[name_loader]
        test_loss, test_MAE, test_ACC = predict(model, loader, None, args, mode=args.test_mode, name_loader=name_loader)
        logging.info(f'Model_best in {args.dataset} dataset group {name_loader}, MAE: {test_MAE}, ACC:{test_ACC}')
    # --------------------------------------


if __name__ == '__main__':
    main()





