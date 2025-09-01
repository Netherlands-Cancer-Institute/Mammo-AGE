import os
import pickle
import heapq
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils import AverageMeter, adjust_learning_rate
from utils.plot_scrtplot import plot3images
from utils.metrics import get_metric


def plot_Scatter(x, y, output_dir='', name='Normal_pred_age'):
    x, y = np.squeeze(x), np.squeeze(y)
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', linestyle='--')
    plt.xlabel('Chronological Age')
    plt.ylabel('Predicted')
    plt.title(name)
    plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', dpi=300)
    plt.close()


def result_analysis_single(args, age_diff, pred_age, age, figure=True, name='B'):
    """
    P value and plot
    """
    # Concatenate and reshape data
    age_diff_np = np.squeeze(np.concatenate(age_diff).reshape(-1, 1))
    abs_age_diff_np = np.abs(age_diff_np)
    pred_age_np = np.squeeze(np.concatenate(pred_age).reshape(-1, 1))
    age_np = np.squeeze(np.concatenate(age).reshape(-1, 1))

    # Calculate mean absolute error (MAE) and mean error (ME) with their standard deviations
    nor_mae, nor_mae_std = np.mean(abs_age_diff_np), np.std(abs_age_diff_np)
    nor_me, nor_me_std = np.mean(age_diff_np), np.std(age_diff_np)

    # Compute Spearman correlation coefficients
    # Age_nor_pearsonr = stats.pearsonr(pred_age_np, age_np)
    Age_nor_spearmanr = stats.spearmanr(pred_age_np, age_np)
    # MAE_nor_pearsonr = stats.pearsonr(abs_age_diff_np, age_np)
    MAE_nor_spearmanr = stats.spearmanr(abs_age_diff_np, age_np)
    # ME_nor_pearsonr = stats.pearsonr(age_diff_np, age_np)
    ME_nor_spearmanr = stats.spearmanr(age_diff_np, age_np)

    # Log the results
    logging.info('===============================TEST================================')
    logging.info('{} dataset {} Group: MAE: {}, MAE_STD: {}, ME:{}, ME_STD:{}'.format(
        args.dataset, name, nor_mae, nor_mae_std, nor_me, nor_me_std))
    logging.info('{} dataset {} Group: Age_spearmanr: {}'.format(args.dataset, name, Age_nor_spearmanr))
    logging.info('{} dataset {} Group: MAE_spearmanr: {}'.format(args.dataset, name, MAE_nor_spearmanr))
    logging.info('{} dataset {} Group: ME_spearmanr: {}'.format(args.dataset, name, ME_nor_spearmanr))
    # logging.info('Normal Group: Age_pearsonr: {}, Age_spearmanr: {}'.format(Age_nor_pearsonr, Age_nor_spearmanr))
    # logging.info('Normal Group: MAE_pearsonr: {}, MAE_spearmanr: {}'.format(MAE_nor_pearsonr, MAE_nor_spearmanr))
    # logging.info('Normal Group: ME_pearsonr: {}, ME_spearmanr: {}'.format(ME_nor_pearsonr, ME_nor_spearmanr))

    # Define output directory for plots
    output_dir = '{}/Predict_{}/'.format(args.results_dir, args.dataset)

    # Plot scatter plots if figure is True
    if figure is True:
        plot_Scatter(age_np, pred_age_np, output_dir=output_dir, name='{}_{}_pred_age'.format(args.dataset, name))
        plot_Scatter(age_np, age_diff_np, output_dir=output_dir, name='{}_{}_gap_age'.format(args.dataset, name))
        plot_Scatter(age_np, abs_age_diff_np, output_dir=output_dir, name='{}_{}_abs_gap_age'.format(args.dataset, name))


# Define a function to compute the loss and metrics
def batch_step(args, input, model, criterion, criterion_density, cal_mae_acc):
    # Define a nested function to compute the loss
    def compute_loss(pred_pathway, emb, log_var, age_label, criterion, use_sto, start_age):
        if criterion is None:
            return torch.zeros(1).cuda()

        criterion1, criterion2 = criterion
        # Compute the ProbOrdiLoss using the first criterion
        _, _, _, ProbOrdiLoss = criterion1(pred_pathway, emb, log_var, age_label - start_age, None, use_sto=use_sto)

        # Adjust the shape of pred_pathway and age_label based on the use_sto flag
        if use_sto:
            class_dim = pred_pathway.shape[-1]
            sample_size = pred_pathway.shape[0]
            pred_pathway_ = pred_pathway.view(-1, class_dim)
            age_label_ = age_label.repeat(sample_size)
        else:
            pred_pathway_ = pred_pathway
            age_label_ = age_label

        # Compute the MeanVarianceLoss using the second criterion
        mv_loss = criterion2(pred_pathway_, age_label_)

        return ProbOrdiLoss + mv_loss

    img, age_label = input['img'].cuda(), input['age'].cuda()
    density_label = input['density'].type(torch.LongTensor).cuda()

    # Get the model predictions and embeddings
    pred_global_pathway, pred_local_pathway, global_emb, local_emb, global_log_var, local_log_var, pred_density = model(
        img, max_t=args.max_t, use_sto=args.use_sto)

    # Compute the density loss
    loss_density = criterion_density(pred_density, density_label)

    # Compute the global loss
    global_loss = compute_loss(pred_global_pathway, global_emb, global_log_var, age_label, criterion, args.use_sto, args.min_age)

    # Initialize the local loss
    local_loss = torch.zeros(1).cuda()

    # Compute the local loss for each local pathway
    for index_ in range(len(pred_local_pathway)):
        local_loss += compute_loss(pred_local_pathway[index_], local_emb[index_], local_log_var[index_], age_label, criterion, args.use_sto, args.min_age)

    # Compute the total loss as the sum of global, local, and density losses
    total_loss = global_loss + local_loss + (loss_density * args.lambda_density)

    # Calculate the mean absolute error (MAE) and accuracy for local and global predictions
    local_result = cal_mae_acc(pred_local_pathway, age_label - args.min_age, args.use_sto, GL_multiview=True)
    global_result = cal_mae_acc(pred_global_pathway, age_label - args.min_age, args.use_sto)

    return total_loss, global_result, global_loss, local_result, local_loss, loss_density, pred_density


def train_loop(model, data_loader, criterion, optimizer, epoch, args):
    # Initialize metrics
    metrics = {
        'losses': AverageMeter(),
        'global_losses': AverageMeter(),
        'local_losses': AverageMeter(),
        'mae': AverageMeter(),
        'global_mae': AverageMeter(),
        'local_mae': AverageMeter(),
        'global_acc': AverageMeter(),
        'local_acc': AverageMeter(),
        'acc': AverageMeter(),
        'density_losses': AverageMeter()
    }
    cal_mae_acc = get_metric(args.main_loss_type)
    criterion_density = nn.CrossEntropyLoss(ignore_index=-1)

    # Set model to training mode
    model.train()

    # Adjust learning rate
    adjust_learning_rate(optimizer, epoch, args)

    # Initialize total loss and number of samples
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for i_iter, input in enumerate(train_bar, 1):
        img = input['img']
        batch_size = img.size(0)

        # Compute loss and results
        loss, global_result, global_loss, local_result, local_loss, loss_density, pred_density_result = batch_step(
            args, input, model, criterion, criterion_density, cal_mae_acc)

        # Update metrics
        metrics['global_acc'].update(global_result['acc'])
        metrics['global_mae'].update(global_result['mae'])
        metrics['local_acc'].update(local_result['acc'])
        metrics['local_mae'].update(local_result['mae'])
        metrics['losses'].update(loss.cpu().data.numpy().item())
        metrics['global_losses'].update(global_loss.cpu().data.numpy().item())
        metrics['local_losses'].update(local_loss.cpu().data.numpy().item())
        metrics['density_losses'].update(loss_density.cpu().data.numpy())
        metrics['mae'].update((local_result['mae'] + global_result['mae']) / 2)
        metrics['acc'].update((local_result['acc'] + global_result['acc']) / 2)

        # Gradient accumulation
        if args.accumulation_steps == 1:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = loss / args.accumulation_steps
            loss.backward()
            if i_iter % args.accumulation_steps == 0 or i_iter == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

        total_num += batch_size
        total_loss += loss.item() * batch_size * args.accumulation_steps

        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, MAE: {:.4f}, ACC: {:.4f}, Loss: {:.4f}'.format(
                epoch, args.epochs, optimizer.param_groups[0]['lr'], metrics['mae'].avg, metrics['acc'].avg, metrics['losses'].avg))

    logging.info('Train Epoch {} : Loss is {}, Global Loss: {}, Local Loss: {}, Density Loss: {}'.format(
        epoch, metrics['losses'].avg, metrics['global_losses'].avg, metrics['local_losses'].avg, metrics['density_losses'].avg))
    logging.info('Min MAE is {}, Global MAE is {}, Local MAE is {}'.format(
        metrics['mae'].avg, metrics['global_mae'].avg, metrics['local_mae'].avg))
    logging.info('Min ACC is {}, Global ACC is {}, Local ACC is {}'.format(
        metrics['acc'].avg, metrics['global_acc'].avg, metrics['local_acc'].avg))

    return total_loss / total_num, min(metrics['global_mae'].avg, metrics['local_mae'].avg), max(metrics['global_acc'].avg, metrics['local_acc'].avg)


def evaluation_loop(model, data_loader, criterion, args,  mode='validate', name_loader=None):
    # Set the model to evaluation mode
    model.eval()
    cal_mae_acc = get_metric(args.main_loss_type)
    criterion_density = nn.CrossEntropyLoss(ignore_index=-1)

    # Initialize metrics
    metrics = {
        'total_mae': 0.0, 'total_acc': 0.0, 'total_num': 0,
        'local_mae': 0.0, 'local_acc': 0.0, 'global_mae': 0.0, 'global_acc': 0.0,
        'losses': AverageMeter(), 'global_losses': AverageMeter(), 'local_losses': AverageMeter(),
        'density_losses': AverageMeter()}

    # Lists to store age differences, predicted ages, and actual ages
    age_diff, pred_age, age = [], [], []
    # Lists to store patient ID, exam ID, and image paths (only for predict mode)
    pid, exam_id, img_path = [], [], []
    # Lists to store predicted ages for each view (only for predict mode)
    pred_age_exam, pred_age_lcc, pred_age_rcc, pred_age_lmlo, pred_age_rmlo = [], [], [], [], []
    # Lists to store max and min predicted ages (only for predict mode)
    pred_age_max, pred_age_min = [], []
    # Lists to store density prediction
    pred_density = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for input in tqdm(data_loader):
            img, age_label = input['img'].cuda(), input['age'].cuda()
            race_label = input['race'].type(torch.LongTensor).cuda()
            density_label = input['density'].type(torch.LongTensor).cuda()
            batch_size = img.size(0)

            # Perform a forward pass and compute the loss and metrics
            loss, global_result, global_loss, local_result, local_loss, loss_density, pred_density_result = batch_step(
                args, input, model, criterion, criterion_density, cal_mae_acc)

            # Calculate mean absolute error (MAE) and accuracy
            batch_mae = (local_result['mae'] + global_result['mae']) / 2
            batch_acc = (local_result['acc'] + global_result['acc']) / 2

            # Append results to lists
            age_diff.append((local_result['error_batch']))
            pred_age.append((local_result['pred_age']))
            pred_age_exam.append((global_result['pred_age']))
            pred_age_lcc.append((local_result['pred_age_0']))
            pred_age_rcc.append((local_result['pred_age_1']))
            pred_age_lmlo.append((local_result['pred_age_2']))
            pred_age_rmlo.append((local_result['pred_age_3']))
            pred_age_max.append((local_result['pred_age_max']))
            pred_age_min.append((local_result['pred_age_min']))
            pred_density.append(nn.functional.softmax(pred_density_result, dim=-1).cpu().numpy())

            age.append(age_label.data.cpu().numpy())

            if mode == 'predict':
                img_path.append([input['img_l_cc_path'], input['img_r_cc_path'],
                                 input['img_l_mlo_path'], input['img_r_mlo_path']])
                pid.append(input['pid'])
                exam_id.append(input['exam_id'])

            # Update metrics
            metrics['total_num'] += batch_size
            metrics['total_mae'] += batch_mae.item() * batch_size
            metrics['total_acc'] += batch_acc.item() * batch_size
            metrics['local_mae'] += local_result['mae'].item() * batch_size
            metrics['local_acc'] += local_result['acc'].item() * batch_size
            metrics['global_mae'] += global_result['mae'].item() * batch_size
            metrics['global_acc'] += global_result['acc'].item() * batch_size
            metrics['losses'].update(loss.cpu().data.numpy().item())
            metrics['global_losses'].update(global_loss.cpu().data.numpy().item())
            metrics['local_losses'].update(local_loss.cpu().data.numpy().item())
            metrics['density_losses'].update(loss_density.cpu().data.numpy())

            # Print progress
            tqdm.write(f"{mode} MAE: {metrics['total_mae'] / metrics['total_num']:.4f}, "
                       f"{mode} ACC: {metrics['total_acc'] / metrics['total_num']:.4f}")

    result_analysis_single(args, age_diff, pred_age, age, figure=False, name=mode)

    if mode == 'predict':
        save_dict = {'pid': pid, 'exam_id': exam_id, 'age': age, 'pred_age': pred_age,  'img_path': img_path,
                     'pred_age_exam': pred_age_exam, 'pred_age_lcc': pred_age_lcc, 'pred_age_rcc': pred_age_rcc,
                     'pred_age_lmlo': pred_age_lmlo, 'pred_age_rmlo': pred_age_rmlo,
                     'pred_age_max': pred_age_max, 'pred_age_min': pred_age_min, 'pred_density': pred_density}
        os.makedirs(f'{args.results_dir}/Predict_{args.dataset}/', exist_ok=True)
        pickle.dump(save_dict, open(f'{args.results_dir}/Predict_{args.dataset}/{name_loader}.pkl', 'wb'))

        plot3images(f'{args.results_dir}/Predict_{args.dataset}/{name_loader}.pkl',
                    f'{args.results_dir}/Predict_{args.dataset}/{name_loader}')

    # Log the final results
    logging.info(f"{mode} Min MAE is {metrics['total_mae'] / metrics['total_num']}, "
                 f"Global MAE is {metrics['global_mae'] / metrics['total_num']}, "
                 f"Local MAE is {metrics['local_mae'] / metrics['total_num']}")
    logging.info(f"{mode} Min ACC is {metrics['total_acc'] / metrics['total_num']}, "
                 f"Global ACC is {metrics['global_acc'] / metrics['total_num']}, "
                 f"Local ACC is {metrics['local_acc'] / metrics['total_num']}")

    return metrics['losses'].avg, metrics['local_mae'] / metrics['total_num'], metrics['local_acc'] / metrics['total_num']


def get_train_val_test_demo(model_method):
    if model_method != "Mammo-AGE":
        raise ValueError("Model method not supported.")
    return train_loop, evaluation_loop


def get_predict_demo(model_method, vis=False):
    if vis:
        return Multiview_predict_vis
    else:
        return evaluation_loop


# def inference(args, model, cal_mae_acc, img, race_label, age_label):
#     if 'POE' in args.model_method:
#         if args.input_race and args.multi_task:
#             pred_global_pathway, pred_local_pathway, \
#             global_emb, local_emb, global_log_var, local_log_var, pred_density = model(
#                 img, race_input=race_label, max_t=args.max_t, use_sto=args.use_sto)
#         elif args.multi_task:
#             pred_global_pathway, pred_local_pathway, \
#             global_emb, local_emb, global_log_var, local_log_var, pred_density = model(
#                 img, race_input=None, max_t=args.max_t, use_sto=args.use_sto)
#         elif args.input_race:
#             pred_global_pathway, pred_local_pathway, \
#             global_emb, local_emb, global_log_var, local_log_var = model(
#                 img, race_input=race_label, max_t=args.max_t, use_sto=args.use_sto)
#         else:
#             pred_global_pathway, pred_local_pathway, \
#             global_emb, local_emb, global_log_var, local_log_var = model(
#                 img, race_input=None, max_t=args.max_t, use_sto=args.use_sto)
#     else:
#         if args.input_race and args.multi_task:
#             pred_global_pathway, pred_local_pathway, pred_density = model(img, race_label)
#         elif args.multi_task:
#             pred_global_pathway, pred_local_pathway, pred_density = model(img)
#         elif args.input_race:
#             pred_global_pathway, pred_local_pathway = model(img, race_label)
#         else:
#             pred_global_pathway, pred_local_pathway = model(img)
#
#     # pred_global_pathway, pred_local_pathway = model(img)
#     if args.main_loss_type == 'cls':
#         local_result = cal_mae_acc(pred_local_pathway, age_label, args.use_sto, GL_multiview=True)
#     else:
#         pred_local_pathway_ = sum(pred_local_pathway) / len(pred_local_pathway)
#         local_result = cal_mae_acc(pred_local_pathway_, age_label, args.use_sto)
#
#     global_result = cal_mae_acc(pred_global_pathway, age_label, args.use_sto)
#
#     batch_mae = (local_result['mae'] + global_result['mae']) / 2
#     batch_acc = (local_result['acc'] + global_result['acc']) / 2
#     # diff_age = (local_result['error_batch'] + global_result['error_batch']) / 2
#     diff_age = (local_result['error_batch'])
#     # pred_age = (local_result['pred_age'] + global_result['pred_age']) / 2
#     pred_age = (local_result['pred_age'])
#
#     return batch_mae, batch_acc, diff_age, pred_age, local_result, global_result


# def comput_delt_mae(args, model, cal_mae_acc, img, race_label, age_label, test_mae):
#     size_pixels = [32, 64, 128, 256]
#     # size_pixels = [128, 256]
#     # size_pixel = 32
#     delt_maes = []
#     for size_pixel in size_pixels:
#         B, N, C, H, W = img.shape
#         # print('B, N, C, H, W:', B, N, C, H, W)
#         delt_mae = torch.zeros(B, 4, H // size_pixel, W // size_pixel)
#         for view in range(4):
#             for x in range(H // size_pixel):
#                 for y in range(W // size_pixel):
#                     new_img = img.clone()
#                     new_img[:, view, :, x * 32:(x + 1) * 32, y * 32:(y + 1) * 32] = -1
#
#                     batch_mae, batch_acc, diff_age, pred_age, local_result, global_result = inference(
#                         args, model, cal_mae_acc, new_img, race_label, age_label)
#
#                     delt_ = local_result['mae_batch'] - test_mae
#                     for b_i in range(B):
#                         delt_mae[b_i,view, x, y] = delt_[b_i]
#                     # delt_mae[view, x, y] = local_result['mae'].item() - test_mae
#         delt_maes.append(delt_mae)
#
#     return delt_maes


# def Multiview_predict_vis(model, normal_test_loader, args, name_loader):
#     model.eval()
#     cal_mae_acc = get_metric(args.main_loss_type)
#     age_diff, pred_age, age, delt_mae_vis = [], [], [], []
#     img_path, pid, exam_id = [], [], []
#
#     i = 0
#     save_dict = {}
#     save_dict_keys = ['age', 'pred_age', 'delt_mae_vis', 'img_path', 'pid', 'exam_id']
#
#     with torch.no_grad():
#         normal_test_bar = tqdm(normal_test_loader)
#         for input in normal_test_bar:
#             i += 1
#             if i % 1 == 0:
#                 for key in save_dict_keys:
#                     save_dict[key] = eval(key)
#                 os.makedirs('{}/Vis_{}/{}/'.format(args.results_dir, args.dataset, name_loader), exist_ok=True)
#                 pickle.dump(save_dict, open('{}/Vis_{}/{}.pkl'.format(args.results_dir, args.dataset, name_loader), 'wb'))
#             if i > 20:
#                 break
#
#             img, age_label = input['img'], input['age']
#             img_path.append([input['img_l_cc_path'], input['img_r_cc_path'],
#                              input['img_l_mlo_path'], input['img_r_mlo_path']])
#
#             pid.append(input['pid'])
#             exam_id.append(input['exam_id'])
#
#             img, age_label = input['img'].cuda(), input['age'].cuda()
#             race_label = input['race'].type(torch.LongTensor)
#             race_label = race_label.cuda()
#             # density_label = input['density'].type(torch.LongTensor)
#             # density_label = density_label.cuda()
#
#             batch_mae, batch_acc, diff_age, pred_age, local_result, global_result = inference(
#                 args, model, cal_mae_acc, img, race_label, age_label)
#
#             age_diff_normal.append(diff_age)
#             pred_age_normal.append(pred_age)
#             age_normal.append(age_label.data.cpu().numpy())
#             normal_total_num += normal_test_loader.batch_size
#             normal_total_mae += batch_mae.item() * normal_test_loader.batch_size
#             normal_total_acc += batch_acc.item() * normal_test_loader.batch_size
#
#             normal_total_local_mae += local_result['mae'].item() * normal_test_loader.batch_size
#             normal_total_local_acc += local_result['acc'].item() * normal_test_loader.batch_size
#
#             normal_total_global_mae += global_result['mae'].item() * normal_test_loader.batch_size
#             normal_total_global_acc += global_result['acc'].item() * normal_test_loader.batch_size
#
#             normal_test_bar.set_description('normal test MAE: {:.4f}, normal test ACC: {:.4f}'.format(
#                 normal_total_mae / normal_total_num, normal_total_acc / normal_total_num,
#             ))
#
#             delt_mae = comput_delt_mae(args, model, cal_mae_acc, img, race_label, age_label, local_result['mae_batch'])
#             delt_mae_vis.append(delt_mae)
#
#
#         logging.info('Normal test Min MAE is {}, Golbal MAE is {}, Local MAE is {}'.format(
#             normal_total_mae / normal_total_num,
#             normal_total_global_mae / normal_total_num,
#             normal_total_local_mae / normal_total_num))
#         logging.info('Normal test Min ACC is {}, Golbal ACC is {}, Local ACC is {}'.format(
#             normal_total_acc / normal_total_num,
#             normal_total_global_acc / normal_total_num,
#             normal_total_local_acc / normal_total_num))
#
#     result_analysis_single(args, age_diff_normal, pred_age_normal, age_normal, figure=False, name=name_loader)
#
#
#     save_dict['age'] = age_normal
#     save_dict['pred_age'] = pred_age_normal
#     save_dict['delt_mae_vis'] = delt_mae_vis
#     save_dict['img_path'] = img_path
#     save_dict['pid'] = pid
#     save_dict['exam_id'] = exam_id
#
#     os.makedirs('{}/Vis_{}/'.format(args.results_dir, args.dataset), exist_ok=True)
#     pickle.dump(save_dict, open('{}/Vis_{}/{}.pkl'.format(args.results_dir, args.dataset, name_loader), 'wb'))
#
#     # plot3images('{}/Predict_{}/{}.pkl'.format(args.results_dir, args.dataset, name_loader),
#     #             '{}/Predict_{}/{}'.format(args.results_dir, args.dataset, name_loader))
#
#     return normal_total_local_mae / normal_total_num, normal_total_local_acc / normal_total_num


def comput_delt_mae(args, model, cal_mae_acc, img, race_label, age_label, density_label, test_mae_batch):
    model.eval()
    criterion_density = nn.CrossEntropyLoss(ignore_index=-1)

    size_pixels, delt_maes = [32, 64, 128, 256], []
    B, N, C, H, W = img.shape

    with torch.no_grad():
        for size_pixel in size_pixels:
            delta_mae_tensor = torch.zeros(B, 4, H // size_pixel, W // size_pixel).cuda()

            for view in range(4):
                for x in range(H // size_pixel):
                    for y in range(W // size_pixel):
                        img_occluded = img.clone()
                        x_start, x_end = x * size_pixel, (x + 1) * size_pixel
                        y_start, y_end = y * size_pixel, (y + 1) * size_pixel
                        img_occluded[:, view, :, x_start:x_end, y_start:y_end] = -1
                        input = {'img': img_occluded, 'age': age_label, 'race': race_label, 'density': density_label}

                        _, _, _, local_result, _, _, _ = batch_step(args, input, model, criterion=None,
                            criterion_density=criterion_density, cal_mae_acc=cal_mae_acc)

                        delta = local_result['mae_batch'] - test_mae_batch
                        # Ensure type is Tensor and on same device
                        if isinstance(delta, np.ndarray):
                            delta = torch.from_numpy(delta).to(delta_mae_tensor.device)
                        else:
                            delta = delta.to(delta_mae_tensor.device)

                        delta_mae_tensor[:, view, x, y] = delta

            delt_maes.append(delta_mae_tensor.cpu())

    return delt_maes


def Multiview_predict_vis(model, data_loader, criterion, args, mode='predict', name_loader=None):
    """
    Multiview prediction + saliency visualization using occlusion-based delta MAE.
    Matches the input/output interface of `evaluation_loop` for seamless interchange.
    """
    model.eval()
    cal_mae_acc = get_metric(args.main_loss_type)
    criterion_density = nn.CrossEntropyLoss(ignore_index=-1)

    # Result buffers
    age_diff, pred_age, age, delt_mae_vis = [], [], [], []
    img_path, pid, exam_id = [], [], []

    total_mae, total_acc = 0.0, 0.0
    total_local_mae, total_local_acc = 0.0, 0.0
    total_global_mae, total_global_acc = 0.0, 0.0
    total_num = 0
    dummy_loss = 0.0  # No loss computed here, return 0.0 for interface compatibility

    name_loader = name_loader or mode  # fallback if None

    for idx, input in enumerate(tqdm(data_loader), 1):
        img, age_label = input['img'].cuda(), input['age'].cuda()
        race_label = input['race'].type(torch.LongTensor).cuda()
        density_label = input['density'].type(torch.LongTensor).cuda()
        bsz = img.size(0)

        # Forward pass
        _, global_result, _, local_result, _, _, _ = batch_step(args, input, model, criterion=criterion,
            criterion_density=criterion_density, cal_mae_acc=cal_mae_acc)

        # Average MAE and accuracy
        batch_mae = (local_result['mae'] + global_result['mae']) / 2
        batch_acc = (local_result['acc'] + global_result['acc']) / 2

        # Accumulate metrics
        total_num += bsz
        total_mae += batch_mae.item() * bsz
        total_acc += batch_acc.item() * bsz
        total_local_mae += local_result['mae'].item() * bsz
        total_local_acc += local_result['acc'].item() * bsz
        total_global_mae += global_result['mae'].item() * bsz
        total_global_acc += global_result['acc'].item() * bsz

        # Collect sample-wise outputs
        age_diff.append(local_result['error_batch'])
        pred_age.append(local_result['pred_age'])
        age.append(age_label.cpu().numpy())
        pid.append(input['pid'])
        exam_id.append(input['exam_id'])
        img_path.append([input['img_l_cc_path'], input['img_r_cc_path'],
                         input['img_l_mlo_path'], input['img_r_mlo_path']])

        # Saliency map: occlusion-based ΔMAE
        delt_mae = comput_delt_mae(
            args, model, cal_mae_acc, img, race_label, age_label, density_label, local_result['mae_batch'])
        delt_mae_vis.append(delt_mae)

        # Save intermediate results
        if idx % 1 == 0:
            save_dict = {'age': age, 'pred_age': pred_age, 'delt_mae_vis': delt_mae_vis,
                'img_path': img_path, 'pid': pid, 'exam_id': exam_id}
            save_dir = f'{args.results_dir}/Vis_{args.dataset}/{name_loader}/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(save_dict, open(f'{save_dir}/{name_loader}.pkl', 'wb'))

        # Limit number of visualized samples
        if idx >= 1000:
            break

    # Compute averages
    avg_local_mae, avg_local_acc = total_local_mae / total_num, total_local_acc / total_num

    # Logging
    logging.info('Visual Test Results - MAE: {:.4f}, Global MAE: {:.4f}, Local MAE: {:.4f}'.format(
        total_mae / total_num, total_global_mae / total_num, avg_local_mae))
    logging.info('Visual Test Results - ACC: {:.4f}, Global ACC: {:.4f}, Local ACC: {:.4f}'.format(
        total_acc / total_num, total_global_acc / total_num, avg_local_acc))

    # Final correlation analysis
    result_analysis_single(args, age_diff, pred_age, age, figure=False, name=name_loader)

    return dummy_loss, avg_local_mae, avg_local_acc  # Matches evaluation_loop return interface






