import os
import shutil
import argparse
import json
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
sys.path.append(os.path.join(current_dir, 'losses'))
sys.path.append(os.path.join(current_dir, 'learning'))
sys.path.append(os.path.join(current_dir, 'models'))
import warnings
warnings.filterwarnings('ignore')
from utils.opts import arg_parse_test, get_criterion, get_model, get_learning_demo, get_hyperparams_for_test, get_dataset
from utils.utils import *
from utils.mylogging import open_log


def get_data_info(args):
    # Read the dataset config file
    dataset_cfg = read_yaml(args.dataset_config)
    logging.info(dataset_cfg)

    base_image_dir = args.image_dir  # get the base image directory

    args.image_dir = []  # initialize the image directory
    for dataset in args.dataset:
        try:
            # Append the image directory for each dataset
            # args.image_dir.append(os.path.join(base_image_dir, dataset_cfg[dataset]['image_dir']))
            args.image_dir.append(f"{base_image_dir}/{dataset_cfg[dataset]['image_dir']}")
        except KeyError:
            raise ValueError(f" DATASET: {dataset} is not supported.")

    datasets = args.dataset
    train_data_info, valid_data_info, test_data_info = [], [], []

    for dataset in args.dataset:
        try:
            # Get the CSV directory for the dataset
            data_info = pd.read_csv(f'{args.csv_dir}/{args.task}/{dataset}_data_info.csv')
            if args.years_at_least_followup != 0:
                data_info = data_info[(data_info['years_to_last_followup'] > args.years_at_least_followup - 1)
                                      | (data_info["years_to_cancer"] != 100)]
                data_info = data_info.reset_index()

            logging.info(f" DATASET: {dataset}")

            _test_data_info = data_info[data_info['split_group'] == 'test']
            _test_data_info = _test_data_info.reset_index()
            logging.info(f" Length of test data: {len(_test_data_info)}")
            logging.info(f" DATASET INFO: {_test_data_info}")

            test_data_info.append(_test_data_info)

        except KeyError:
            raise ValueError(f" DATASET: {dataset} is not supported.")

    return datasets, test_data_info


def test():
    args = arg_parse_test()
    args = get_hyperparams_for_test(args)
    seed_reproducer(seed=args.seed)
    dataset_name = args.dataset[0]

    open_log(args, name=f'{dataset_name}_test')
    logging.info(str(args).replace(',', "\n"))

    # Data prepare
    # ---------------------------------
    datasets_names, test_data_info = get_data_info(args)
    custom_dataset, dataloador = get_dataset(args)

    # Data loader
    # ---------------------------------
    test_loader = dataloador(custom_dataset, test_data_info, datasets_names, args)
    logging.info('finish data loader')

    # define criterion
    # ----------------------------------------
    criterion = get_criterion(args)

    # define Model
    # ----------------------------------------
    model = get_model(args)

    # define train val test demo
    # ----------------------------------------
    _, _, test = get_learning_demo(args)
    # ---------------------------------

    # ===========  load weights ===========  #
    logging.info("=> loading checkpoint '{}'".format(args.path_risk_model + '/model_best.pth.tar'))
    checkpoint = torch.load(args.path_risk_model + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # ---------------------------------

    # ===========  evaluate final best model on test set ===========  #
    # test_result = test(model, test_loader, criterion, args, save_pkl=f'predict_{dataset_name}', inference=True)
    test_result = test(model, test_loader, criterion, args, save_pkl=f'predict_{dataset_name}', inference=False)

    logging.info('Final test Model_best in {} test dataset, MAE: {}, ACC:{}, C-index: {}'.format(
        dataset_name, test_result['mae'], test_result['acc'], test_result['c_index']))


if __name__ == '__main__':
    test()


