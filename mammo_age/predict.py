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


def get_test_data_info_lists(subsets, _csv, fold):
    test_data_info_lists = {}
    for i in range(len(subsets)):
        subset = subsets[i]
        if subset == 'train' or subset == 'valid':
            _data_info = pd.read_csv(f'{_csv}/{subset}_data_info_{fold}_fold.csv')
        else:
            _data_info = pd.read_csv(f'{_csv}/{subset}_data_info.csv')

        test_data_info_lists[subset] = _data_info

    return test_data_info_lists


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


# TODO: Need to revise this function by adding configs and deleting the private paths in the code
def get_data_info(args):
    def get_patient_list(all_patient_id):
        all_patient_id = list(all_patient_id)
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        return patient_id

    _csv = '/projects/xin-aging/proj-breast-aging/codes/breast_age/CSV/{}-CSV'.format(args.dataset)
    if args.dataset == 'CMMD':
        args.image_dir = ["/projects/mammogram_data/CMMD-data/preprocessed_CMMD_20230506"]
        # args.image_dir = ["/processing/x.wang/CMMD-data/preprocessed_CMMD_20230506"]
        args.csv_dir = "/projects/mammogram_data/CMMD-data/CMMD_clinicaldata_revision.csv"
        all_data_info = pd.read_csv(args.csv_dir)
        Malignant_data_info_ = all_data_info[(all_data_info['classification'] == 'Malignant')]

        patient_ids = get_patient_list(all_data_info['ID1'])
        Malignant_pids = get_patient_list(Malignant_data_info_['ID1'])

        Benign_pids = []
        for i_pid in range(len(patient_ids)):
            pid_ = patient_ids[i_pid]
            if pid_ not in Malignant_pids:
                Benign_pids.append(pid_)

        index_ = [i for (i, v) in enumerate(list(all_data_info["ID1"]))
                  for x in range(0, len(Benign_pids))
                  if v == Benign_pids[x]]
        Benign_data_info = all_data_info.iloc[index_, :]

        index_ = [i for (i, v) in enumerate(list(all_data_info["ID1"]))
                  for x in range(0, len(Malignant_pids))
                  if v == Malignant_pids[x]]
        Malignant_data_info = all_data_info.iloc[index_, :]

        subsets = [
            'all',
            'Benign',
            'Malignant', ]
        test_data_info_lists = {
            'all': all_data_info,
            'Benign': Benign_data_info,
            'Malignant': Malignant_data_info,
        }
    elif args.dataset == 'BMCD':
        args.image_dir = ["/projects/mammogram_data/BMCD-data/BMCD/preprocessed_BMCD"]
        args.csv_dir = "/projects/mammogram_data/BMCD-data/BMCD_matedata.csv"
        data_info = pd.read_csv(args.csv_dir)
        Normal_data_info = data_info[(data_info['folder'] == 'Normal_cases')]
        Suspicious_data_info = data_info[(data_info['folder'] == 'Suspicious_cases')]
        Benign_data_info = Suspicious_data_info[
            (Suspicious_data_info['classificiation'] == 'benign')
        ]
        Malignant_data_info = Suspicious_data_info[
            (Suspicious_data_info['classificiation'] == 'malignant')]
        subsets = [
            'all',
            'Normal',
            'Suspicious',
            'Benign',
            'Malignant',
        ]
        test_data_info_lists = {
            'all': data_info,
            'Normal': Normal_data_info,
            'Suspicious': Suspicious_data_info,
            'Benign': Benign_data_info,
            'Malignant': Malignant_data_info,
        }
    elif args.dataset == 'EMBED':
        # args.image_dir = ["/data/groups/aiforoncology/archive/radiology/nki-breast/Mammo/EMBED-mammo/preprocessed-img-2048"]
        args.image_dir = ["/processing/x.wang/embed_dataset/preprocessed-img-2048"]
        args.csv_dir = "/projects/xin-aging/proj-breast-aging/codes/breast_age/CSV/EMBED-CSV/" \
                       "external_dataset_MTPBCR_finetune.csv"
        # "external_dataset_MTPBCR_finetune_re730.csv"
        data_info = pd.read_csv(args.csv_dir)
        path_a = '/projects/xin-data/embed_dataset/preprocessed-img-2048'
        path_b = '/processing/x.wang/embed_dataset/preprocessed-img-2048'
        # path_b = '/data/groups/aiforoncology/archive/radiology/nki-breast/Mammo/EMBED-mammo/preprocessed-img-2048'
        data_info['PATH_R_CC'] = data_info['PATH_R_CC'].map(lambda x: x.replace(path_a, path_b))
        data_info['PATH_L_CC'] = data_info['PATH_L_CC'].map(lambda x: x.replace(path_a, path_b))
        data_info['PATH_R_MLO'] = data_info['PATH_R_MLO'].map(lambda x: x.replace(path_a, path_b))
        data_info['PATH_L_MLO'] = data_info['PATH_L_MLO'].map(lambda x: x.replace(path_a, path_b))
        logging.info(data_info.PATH_L_MLO.head(n=10))
        subsets = ['all']
        test_data_info_lists = {'all': data_info, }
    elif args.dataset == 'CESM':
        args.image_dir = ["/projects/mammogram_data/CDD-CESM-data/preprocessed_CDD-CESM/"]
        args.csv_dir = "/projects/mammogram_data/CDD-CESM-data/CDD-CESM-matedata.csv"
        all_data_info = pd.read_csv(args.csv_dir)

        Abnormal_data_info_ = all_data_info[
            (all_data_info['Pathology'] == 'Benign')
            | (all_data_info['Pathology'] == 'Malignant')
            ]
        Malignant_data_info_ = all_data_info[(all_data_info['Pathology'] == 'Malignant')]
        patient_ids = get_patient_list(all_data_info['Patient_ID'])
        Abnormal_pids = get_patient_list(Abnormal_data_info_['Patient_ID'])
        Malignant_pids = get_patient_list(Malignant_data_info_['Patient_ID'])

        Normal_pids = []
        for i_pid in range(len(patient_ids)):
            pid_ = patient_ids[i_pid]
            if pid_ not in Abnormal_pids:
                Normal_pids.append(pid_)
        index_ = [i for (i, v) in enumerate(list(all_data_info["Patient_ID"]))
                  for x in range(0, len(Normal_pids))
                  if v == Normal_pids[x]]
        Normal_data_info = all_data_info.iloc[index_, :]

        Benign_pids = []
        for i_pid in range(len(Abnormal_pids)):
            pid_ = Abnormal_pids[i_pid]
            if pid_ not in Malignant_pids:
                Benign_pids.append(pid_)
        index_ = [i for (i, v) in enumerate(list(all_data_info["Patient_ID"]))
                  for x in range(0, len(Benign_pids))
                  if v == Benign_pids[x]]
        Benign_data_info = all_data_info.iloc[index_, :]

        index_ = [i for (i, v) in enumerate(list(all_data_info["Patient_ID"]))
                  for x in range(0, len(Malignant_pids))
                  if v == Malignant_pids[x]]
        Malignant_data_info = all_data_info.iloc[index_, :]

        subsets = [
            'all',
            'Normal',
            'Benign',
            'Malignant', ]
        test_data_info_lists = {
            'all': all_data_info,
            'Normal': Normal_data_info,
            'Benign': Benign_data_info,
            'Malignant': Malignant_data_info,
        }
    elif args.dataset == 'NKI':
        args.image_dir = ["/projects/mammogram_data/NKI-data-20220323/imgs/"]
        # args.csv_dir = "/processing/x.wang/NKI-data-20220323/data_info_20220320_all.csv"
        # args.csv_dir = "/processing/x.wang/NKI-data-20220323/data_info_20220616_all.csv"
        subsets = [
            'all',
            'train',
            'valid',
            'test',
            'abnormal',
            'abnormal_patient',
            'cancer',
            'diagnosed',
            'high_risk',
            'interval_cancer',
        ]
        test_data_info_lists = get_test_data_info_lists(subsets, _csv, args.fold)
    elif args.dataset == 'Inhouse':
        args.image_dir = [
            "/data/groups/aiforoncology/archive/radiology/nki-breast/Mammo/inhouse/MG/preprocessed/img-2048/"]
            # "/processing/x.wang/NKI-data/preprocessed/img-2048/"]
        subsets = [
            'all',
            'train',
            'valid',
            'test',
            'abnormal',
            'abnormal_patient',
            'cancer',
            'diagnosed',
            'high_risk',
            'interval_cancer',
        ]
        test_data_info_lists = get_test_data_info_lists(subsets, _csv, args.fold)
    elif args.dataset == 'RSNA' or args.dataset == 'VinDr' or args.dataset == 'DDSM':
        if args.dataset == 'RSNA':
            args.image_dir = ["/processing/x.wang/RSNA-2022-Mammo/preprocessing/train"]
            # args.image_dir = ["/projects/mammogram_data/RSNA-2022-Mammo/preprocessing/train"]
            # args.csv_dir = "/projects/mammogram_data/RSNA-2022-Mammo/train.csv"
        elif args.dataset == 'VinDr':
            # args.image_dir = ["/processing/x.wang/vindr-mammo/preprocessing"]
            args.image_dir = ["/projects/mammogram_data/vindr-mammo/preprocessing_old"]
            # args.csv_dir = "/projects/mammogram_data/vindr-mammo/finding_annotations.csv"
        elif args.dataset == 'DDSM':
            args.image_dir = ["/projects/mammogram_data/DDSM-data/preprocessed_DDSM/rerize_1024_512_without_seg"]
        # loaders = {}
        subsets = [
            'all',
            'normal',
            'train',
            'valid',
            'test',
            'benign',
            'cancer',
        ]
        test_data_info_lists = get_test_data_info_lists(subsets, _csv, args.fold)
    else:
        raise f'{args.dataset} not support!!!!'

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





