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
from utils.opts import arg_parse, get_criterion, get_model, get_learning_demo, get_optimizer, get_dataset
from utils.utils import *
from utils.mylogging import open_log
from utils import backup


def get_fraction_data_info(data_info, fraction=0.1, seed=42):
    """
    Get a fraction of the data info based on patient id.
    """

    # Get the unique patient IDs
    unique_patient_ids = data_info['patient_id'].astype(str).unique()

    # Randomly select a fraction of the unique patient IDs use sklearn with the given seed used for reproducibility
    from sklearn.model_selection import train_test_split
    selected_patient_ids, _ = train_test_split(unique_patient_ids, train_size=fraction, random_state=seed)

    # Filter the data_info DataFrame to include only the selected patient IDs
    data_info = data_info[data_info['patient_id'].astype(str).isin(selected_patient_ids)]
    return data_info


def get_screening_data_info(data_info):
    data_info = data_info[
        (data_info['birads'] == 1) | (data_info['birads'] == 2)
        | (data_info['birads'] == 3)
        | (data_info['birads'] == -1) | (data_info['birads'] == 0)
    ]
    return data_info


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

            _train_data_info = data_info[data_info['split_group'] == 'train']
            if args.screening_clean:
                _train_data_info = get_screening_data_info(_train_data_info)

            if args.fraction:
                _train_data_info = get_fraction_data_info(_train_data_info, args.fraction, args.seed)
                logging.info(f" DATASET: {dataset}, fraction: {args.fraction}")

            _train_data_info = _train_data_info.reset_index()
            _valid_data_info = data_info[data_info['split_group'] == 'valid']
            if args.screening_clean:
                _valid_data_info = get_screening_data_info(_valid_data_info)

            _valid_data_info = _valid_data_info.reset_index()
            _test_data_info = data_info[data_info['split_group'] == 'test']
            _test_data_info = _test_data_info.reset_index()

            _train_data_info.to_csv(args.results_dir + f'/{dataset}_train_data_info.csv')
            _valid_data_info.to_csv(args.results_dir + f'/{dataset}_valid_data_info.csv')
            _test_data_info.to_csv(args.results_dir + f'/{dataset}_test_data_info.csv')

            train_data_info.append(_train_data_info)
            valid_data_info.append(_valid_data_info)
            test_data_info.append(_test_data_info)

        except KeyError:
            raise ValueError(f" DATASET: {dataset} is not supported.")

    return datasets, train_data_info, valid_data_info, test_data_info


def main():
    args = arg_parse()
    seed_reproducer(seed=args.seed)

    open_log(args)
    logging.info(str(args).replace(',', "\n"))

    # Training settings
    # ---------------------------------
    best_c_index = 0.0

    # Data prepare
    # ---------------------------------
    datasets_names, train_data_info, valid_data_info, test_data_info = get_data_info(args)
    custom_dataset, dataloador = get_dataset(args)

    # Data loader
    # ---------------------------------
    # args.image_dir = [args.image_dir]
    train_loader = dataloador(custom_dataset, train_data_info, datasets_names, args, train=True)
    valid_loader = dataloador(custom_dataset, valid_data_info, datasets_names, args)
    test_loader = dataloador(custom_dataset, test_data_info, datasets_names, args)
    logging.info('finish data loader')

    # --------------------------------------
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    # define Model and loss
    # ----------------------------------------
    logging.info('finish data loader')

    # define criterion
    # ----------------------------------------
    criterion = get_criterion(args)

    # define Model
    # ----------------------------------------
    model = get_model(args)

    # define train val test demo
    # ----------------------------------------
    train, validate, test = get_learning_demo(args)
    # ---------------------------------

    # define optimizer
    # ---------------------------------
    optimizer = get_optimizer(args, model)

    epoch_start = args.start_epoch
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=1, mode='max', patience=args.lr_decay_patient, factor=0.5)
    early_stopping = EarlyStopping(patience=args.early_stop_patient, verbose=True, mode='max')
    set_backup(args.results_dir)
    # load resume model
    # ----------------------------------------
    if args.resume_retrain:
        checkpoint = torch.load(args.resume_retrain)
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        best_c_index = checkpoint['best_c_index']
        logging.info('Loaded from: {}'.format(args.resume))
        test_result = checkpoint['test_result'] if 'test_result' in checkpoint else None
    # ---------------------------------

    cudnn.benchmark = True
    for epoch in range(epoch_start, args.epochs + 1):
        # =========== start train =========== #
        if epoch == 0:
            # =========== zero shoot test with initial weights =========== #
            # valid_result = validate(model, valid_loader, criterion, args)
            # logging.info('epoch: {}: valid_loss: {}, valid_MAE: {}, valid_ACC:{}, C-index: {}'.format(
            #     epoch, valid_result['loss'], valid_result['mae'], valid_result['acc'], valid_result['c_index']))

            test_result = test(model, test_loader, criterion, args)
            logging.info(
                'Init pretrained Model in val dataset, loss : {}, MAE: {}, ACC: {}, C-index: {}'.format(
                    test_result['loss'], test_result['mae'], test_result['acc'], test_result['c_index']))

        else:
            # ===========  train for one epoch   =========== #
            train_result = train(model, train_loader, criterion, optimizer, epoch, args)
            logging.info('epoch: {}: train_loss: {}, train_MAE: {}, train_ACC:{}, C-index: {}'.format(
                epoch, train_result['loss'], train_result['mae'], train_result['acc'], train_result['c_index']))

            # ===========  evaluate on validation set ===========  #
            valid_result = validate(model, valid_loader, criterion, args)
            logging.info('epoch: {}: valid_loss: {}, valid_MAE: {}, valid_ACC:{}, C-index: {}'.format(
                epoch, valid_result['loss'], valid_result['mae'], valid_result['acc'], valid_result['c_index']))

            # ===========  learning rate decay =========== #
            scheduler.step(valid_result['c_index'])
            for param_group in optimizer.param_groups:
                print("\n*learning rate {:.2e}*\n".format(param_group['lr']))

            # ===========  record the  best metric and save checkpoint ===========  #
            is_best = valid_result['c_index'] > best_c_index
            best_c_index = max(valid_result['c_index'], best_c_index)

            if is_best:
                # ===========  evaluate on test set ===========  #
                test_result = test(model, test_loader, criterion, args, save_pkl=f'best_{epoch}')
                # print('P_value', P_value)
                logging.info(
                    'epoch: {} is test best now, Model_best in test dataset, MAE: {}, ACC:{}, C-index: {}'.format(
                        epoch, test_result['mae'], test_result['acc'], test_result['c_index']))

            save_checkpoint(args.results_dir, {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'best_c_index': best_c_index, 'optimizer': optimizer.state_dict(),
                 'train_result': train_result, 'valid_result': valid_result, 'test_result': test_result,}, is_best)

            # ===========  early_stopping needs the validation loss or MAE to check if it has decresed
            early_stopping(valid_result['c_index'])
            if early_stopping.early_stop:
                logging.info("======= Early stopping =======")
                break

    # ===========  evaluate final best model on test set ===========  #
    checkpoint = torch.load(args.results_dir + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_result = test(model, test_loader, criterion, args, save_pkl='best')

    logging.info('Final test Model_best in test dataset, MAE: {}, ACC:{}, C-index: {}'.format(
        test_result['mae'], test_result['acc'], test_result['c_index']))


def set_backup(custom_backup_dir="custom_backups"):
    custom_backup_dir = os.path.join(custom_backup_dir, "mammo_age_finetune")
    # Save backup of the current script
    backup.save_script_backup(__file__, custom_backup_dir)
    # Backup all imported modules within the project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    backup.backup_imported_modules(project_root, custom_backup_dir)


if __name__ == '__main__':
    main()
