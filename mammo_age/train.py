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
sys.path.append(os.path.join(current_dir, 'losses'))
sys.path.append(os.path.join(current_dir, 'learning'))
sys.path.append(os.path.join(current_dir, 'models'))
# sys.path.append(os.path.join(current_dir, 'utils'))
from utils.utils import *
from utils.mylogging import open_log
from learning.train_val_test_demo import get_train_val_test_demo
from utils import backup
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    # Define the parser
    parser = argparse.ArgumentParser(description='Mammo-AGE')

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

    # Training parameters
    # ---------------------------------
    parser.add_argument('--seed', default=5, type=int, help='seed for initializing training.')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer in [SGD, Adam, RMSprop].')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[20, 40, 60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, metavar='N', help='gradient accumulation steps')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W', help='weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--num-workers', default='16', type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--second_stage', action='store_true', default=False, help='If true, will need to load the weights trained before.')
    parser.add_argument('--first_stage_model', type=str, help='path to latest checkpoint, loading for the second stage training.')
    parser.add_argument('--fold', default=0, type=int, help='5-fold cross validation')

    # Dataset
    # ---------------------------------
    parser.add_argument('--dataset', default=['VinDr'], type=str, nargs='+', help='dataset in [RSNA, VinDr, DDSM, EMBED].')
    parser.add_argument('--img-size', default=1024, type=int, help='size of image')
    parser.add_argument('--dataset-config', default='/Path/to/dataset/config.yaml', type=str, metavar='PATH', help='path to dataset config')
    parser.add_argument('--csv-dir', default='/Path/to/data/base_csv_folder', type=str, metavar='PATH', help='path to base folder for all datasets csv files')
    parser.add_argument('--image-dir', default='/Path/to/data/base_image_folder', type=str, metavar='PATH', help='path to base folder for all datasets image files')

    # Checkpoint
    # ---------------------------------
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # Loss function parameters for Mammo-AGE model
    # ---------------------------------
    parser.add_argument('--loss-method', default='ce', type=str, help='loss type in [l1, mse, ce].')
    parser.add_argument('--main-loss-type', default='cls', type=str, help='loss type in [cls, reg, rank].')
    parser.add_argument('--lambda_density', default=0.2, type=float, metavar='M', help='lambda of density classification loss (default: 0.2)')

    # For POE method
    # ---------------------------------
    parser.add_argument('--max-t', type=int, default=50, help='number of samples during sto.')
    parser.add_argument('--no-sto', action='store_true', default=False, help='not using stochastic sampling when training or testing.')
    parser.add_argument('--distance', type=str, default='JDistance', help='distance metric between two gaussian distribution')
    parser.add_argument('--alpha-coeff', type=float, default=1e-5, metavar='M', help='alpha_coeff (default: 0)')
    parser.add_argument('--beta-coeff', type=float, default=1e-4, metavar='M', help='beta_coeff (default: 1.0)')
    parser.add_argument('--margin', type=float, default=5, metavar='M', help='margin (default: 1.0)')
    # ---------------------------------

    args = parser.parse_args()

    # Set the path to save the results
    args.results_dir = f"{args.results_dir}/{'_'.join(args.dataset)}/{str(args.arch)}/{args.img_size}_fold{args.fold}/"

    if args.resume:
        args.results_dir = args.resume
        with open(args.results_dir + '/args.json', 'r') as f:
            raw_dict = json.load(f)

        args.arch = raw_dict['arch']
        args.main_loss_type = raw_dict['main_loss_type']
        args.loss_method = raw_dict['loss_method']
        args.model_method = raw_dict['model_method']
        args.num_output_neurons = raw_dict['num_output_neurons']
        args.fold = raw_dict['fold']
        args.nblock = raw_dict['nblock']
        args.hidden_size = raw_dict['hidden_size']
        args.seed = raw_dict['seed']
        args.no_sto = raw_dict['no_sto']
        args.dataset = raw_dict['dataset']
        args.batch_size = raw_dict['batch_size']
        args.lr = raw_dict['lr']
        args.img_size = raw_dict['img_size']

    args.use_sto = True if not args.no_sto else False
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def get_dataset(args):
    # Function to get the dataset based on the provided arguments
    dataset_cfg = read_yaml(args.dataset_config) # read the dataset config file
    base_image_dir = args.image_dir # get the base image directory

    args.image_dir = [] # initialize the image directory
    for dataset in args.dataset:
        try:
            # Append the image directory for each dataset
            args.image_dir.append(os.path.join(base_image_dir, dataset_cfg[dataset]['image_dir']))
        except KeyError:
            raise ValueError(f" DATASET: {dataset} is not supported.")

    from dataload.combine_data_loader import dataloador
    from dataload.custom_dataset_4views import CustomDataset
    return CustomDataset, dataloador


def get_data_info(args):
    # Read the dataset config file
    dataset_cfg = read_yaml(args.dataset_config)
    logging.info(dataset_cfg)
    base_csv_dir = args.csv_dir

    datasets = args.dataset
    train_data_info = []
    valid_data_info = []
    test_data_info = []

    for dataset in args.dataset:
        try:
            # Get the CSV directory for the dataset
            _csv = os.path.join(base_csv_dir, dataset_cfg[dataset]['csv_dir'])
            logging.info(f" DATASET: {dataset}: CSV folder: {_csv}")
            train_data_info.append(pd.read_csv(f'{_csv}/train_data_info_{args.fold}_fold.csv'))
            valid_data_info.append(pd.read_csv(f'{_csv}/valid_data_info_{args.fold}_fold.csv'))
            test_data_info.append(pd.read_csv(f'{_csv}/test_data_info.csv'))
        except KeyError:
            raise ValueError(f" DATASET: {dataset} is not supported.")

    return datasets, train_data_info, valid_data_info, test_data_info


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


def main():
    # Parse command-line arguments
    # --------------------------------------
    args = arg_parse()
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # Set the random seed
    # --------------------------------------
    seed_reproducer(seed=args.seed)

    # Set up the logger
    # --------------------------------------
    open_log(args)
    logging.info(str(args).replace(',', "\n"))

    # Data prepare
    # ---------------------------------
    custom_dataset, dataloador  = get_dataset(args)
    datasets_names, train_data_info, valid_data_info, test_data_info = get_data_info(args)

    # Data loader
    # ---------------------------------
    train_loader = dataloador(custom_dataset, train_data_info, datasets_names, args, train=True)
    valid_loader = dataloador(custom_dataset, valid_data_info, datasets_names, args)
    test_loader = dataloador(custom_dataset, test_data_info, datasets_names, args)
    logging.info('finish data loader')

    # Define Model and loss functions
    # ----------------------------------------
    model, criterion = get_model(args)
    logging.info(model)

    # Define train val test demo
    # ----------------------------------------
    train, validate = get_train_val_test_demo(args.model_method)

    # Define optimizer
    # ---------------------------------
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer: {args.optimizer} is not supported.")

    # Set backup
    # --------------------------------
    set_backup(args.results_dir)

    # Training settings
    # ---------------------------------
    epoch_start = args.start_epoch # start from epoch 1
    best_MAE = 500000  # best MAE
    cudnn.benchmark = True # improve the efficiency of the model
    log_results = {'epoch': [], 'train_loss': [], 'train_MAE': [], 'train_ACC': [],
                   'valid_loss': [], 'valid_MAE': [], 'valid_ACC': []}

    # load resume model
    # ----------------------------------------
    if args.resume:
        checkpoint = torch.load(args.resume+ '/model_last.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        best_MAE = checkpoint['best_MAE']
        log_results = checkpoint['log_results']
        logging.info('Loaded from: {}'.format(args.resume))

    # Training loop
    # ---------------------------------
    early_stop = 0
    for epoch in range(epoch_start, args.epochs + 1):
        if epoch == 0:
            valid_loss, MAE, ACC = validate(model, valid_loader, criterion, args, mode='validate')
            logging.info(f'Init pretrained Model in val dataset, loss : {valid_loss}, MAE: {MAE}, ACC: {ACC}')
            test_loss, test_MAE, test_ACC = validate(model, test_loader, criterion, args, mode='test')
            logging.info(f'Init pretrained Model in test dataset, normal_MAE: {test_MAE}, normal_ACC:{test_ACC}')
        else:
            log_results['epoch'].append(epoch)
            train_loss, train_MAE, train_ACC = train(model, train_loader, criterion, optimizer, epoch, args)
            log_results['train_loss'].append(train_loss)
            log_results['train_MAE'].append(train_MAE)
            log_results['train_ACC'].append(train_ACC)
            logging.info(f'epoch: {epoch}: train_loss: {train_loss}, train_MAE: {train_MAE}, train_ACC:{train_ACC}')

            valid_loss, valid_MAE, valid_ACC = validate(model, valid_loader, criterion, args, mode='validate')
            log_results['valid_loss'].append(valid_loss)
            log_results['valid_MAE'].append(valid_MAE)
            log_results['valid_ACC'].append(valid_ACC)
            logging.info(f'epoch: {epoch}: valid_loss: {valid_loss}, valid_MAE: {valid_MAE}, valid_ACC:{valid_ACC}')

            # save log
            data_frame = pd.DataFrame(data=log_results,)
            data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')

            is_best = valid_MAE < best_MAE
            best_MAE = min(valid_MAE, best_MAE)

            if is_best:
                early_stop = 0
                logging.info('epoch: {} is test best now, MAE is : {}'.format(epoch, valid_MAE))
            else:
                early_stop += 1

            # save model
            if is_best or epoch > 8: # save the model after 8 epochs
                save_checkpoint(args.results_dir, {
                    'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_MAE': best_MAE,
                    'optimizer': optimizer.state_dict(), 'log_results': log_results,}, is_best)

            if early_stop > 10:
                logging.info('Early stopping at epoch: {}'.format(epoch))
                break

    # Test the model on the test dataset
    checkpoint = torch.load(args.results_dir + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_MAE, test_ACC = validate(model, test_loader, criterion, args, mode='test')
    #
    logging.info(f'Model_best in test dataset, normal_MAE: {test_MAE}, normal_ACC:{test_ACC}')


# Backup the current script and imported modules
def set_backup(custom_backup_dir="custom_backups"):
    custom_backup_dir = os.path.join(custom_backup_dir, "mammo_age")
    # Save backup of the current script
    backup.save_script_backup(__file__, custom_backup_dir)
    # Backup all imported modules within the project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    backup.backup_imported_modules(project_root, custom_backup_dir)


if __name__ == '__main__':
    main()





