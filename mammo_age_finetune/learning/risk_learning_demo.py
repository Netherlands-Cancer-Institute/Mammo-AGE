import gc
import pickle
import torch
from torch.nn.utils import clip_grad_norm_
from utils.utils import *
from utils.compute_auc_c_index import compute_auc_cindex
from tqdm import tqdm
import numpy as np
import logging
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from torchvision.transforms import v2 as T


def cal_mae_acc(logits, targets, years_last_followup, threshold=1, weights=None):
    """
    Calculates the mean absolute error (MAE) and accuracy (ACC) between model predictions and target labels.

    The function evaluates the performance of a model for tasks involving age prediction by comparing
    predicted probabilities and corresponding target labels. Weighting of errors and accuracy computation
    is supported where applicable. Results include batch-wise errors and other intermediate calculations.

    Parameters:
        logits (torch.Tensor): Predicted logits from the model with shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels for the samples.
        years_last_followup (torch.Tensor): Additional label information indicating years since last follow-up.
        threshold (int, optional): Accuracy threshold, defaults to 1.
        weights (numpy.ndarray or None, optional): Optional weights for the samples. Defaults to None.

    Returns:
        dict: A dictionary containing the following keys:
            - 'mae': Mean Absolute Error computed over valid samples.
            - 'acc': Accuracy computed based on the threshold.
            - 'mae_batch': A batch-wise list of absolute errors.
            - 'error_batch': A batch-wise list of errors.
            - 'pred_age': Predicted ages for valid samples.
            - 'pred_age_all': Predicted ages for all samples.
            - 'count': Number of valid samples used for computation.
    """

    s_dim, out_dim = logits.shape
    probs = F.softmax(logits, -1)
    probs_data = probs.cpu().data.numpy()
    label_arr = np.array(range(out_dim))
    exp_data_all = np.sum(probs_data * label_arr, axis=-1)

    target_data = targets.cpu().data.numpy()
    years_last_followup_data = years_last_followup.cpu().data.numpy()
    # label_arr = np.array(range(out_dim))
    # exp_data_all = np.sum(probs_data * label_arr, axis=1)
    target_data[target_data > (out_dim - 1)] = out_dim - 1
    mask = 1 - ((target_data == (out_dim - 1)) & (years_last_followup_data < (out_dim - 1))).astype(int)

    count = sum(mask)
    if count != 0:
        exp_data_all_ = exp_data_all[mask==1]
        target_data_ = target_data[mask==1]

        error_batch = exp_data_all_ - target_data_
        mae_batch = abs(error_batch)
        if weights is not None:
            weights = np.asarray(weights).reshape(1,-1)
            weights_ = np.repeat(weights, count, axis=0)
            weights_ = weights_[range(count), target_data_]
            # mae = sum(mae_batch * weights_) / sum(weights_)
            # acc = sum((np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * weights_) * 1.0 / sum(weights_)
            mae = np.mean(mae_batch * weights_)
            acc = np.mean((np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * weights_) * 1.0
        else:
            mae = sum(mae_batch) / len(target_data_)
            acc = sum(np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * 1.0 / len(target_data_)
    else:
        mae = 0
        acc = 0
        mae_batch = []
        error_batch = []
        exp_data_all_ = []

    return {
        'mae': mae,
        'acc': acc,
        'mae_batch': mae_batch,
        'error_batch': error_batch,
        'pred_age': exp_data_all_,
        'pred_age_all': exp_data_all,
        'count': count
    }


def compute_losses(args, criterion, input, output, **kwargs):
    """
    Computes the loss for the given input and output.
    """
    # Get the risk label and years to last follow-up from input
    risk_label_ = input['years_to_cancer'].cuda()
    years_last_followup_ = input['years_to_last_followup'].cuda()
    risk = output['risk']

    # Compute basic BCE loss
    criterion_BCE = criterion['criterion_BCE']
    risk_label = risk_label_.clone()
    years_last_followup = years_last_followup_.clone()
    loss = criterion_BCE(risk, risk_label, years_last_followup, weights=args.time_to_events_weights)

    # Add additional loss if present in output
    if output['loss'] is not None:
        loss += output['loss']
    return loss


def input_and_output(args, model, input, **kwargs):
    img = input['img'].cuda()

    # Get the model predictions and embeddings
    output_dict = model(img, max_t=args.max_t, use_sto=args.use_sto)

    # Extract the loss from the output dictionary if it exists
    loss = output_dict['loss'] if 'loss' in output_dict else None

    # Return the predictions, loss, and optionally the output dictionary for inference
    return {'risk': output_dict['predict'], 'loss': loss,
            'output_dict': output_dict if 'inference' in kwargs and kwargs['inference'] else None,}


def direct_train(model, data_loader, criterion, optimizer, epoch, args):
    """

    """

    losses = AverageMeter()
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    i_debug = 0
    for input in train_bar:
        if i_debug > 30 and args.debug:
            break
        i_debug += 1
        # img = input['img']
        if args.debug:
            pickle.dump(input, open('{}/result_{}.pkl'.format(args.results_dir, i_debug), 'wb'))

        output = input_and_output(args, model, input, train=True)
        loss = compute_losses(args, criterion, input, output)

        if args.accumulation_steps == 1:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
            optimizer.step()
        else:
            # Accumulative gradients
            loss = loss / args.accumulation_steps
            loss.backward()
            if i_debug % args.accumulation_steps == 0 or i_debug == len(data_loader):
                # Gradient clipping
                clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
                optimizer.step()
                # Reset gradients, for the next accumulated batches
                optimizer.zero_grad()

        #####################################
        # optimizer.zero_grad()
        # loss.backward()
        # # Gradient clipping
        # clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
        # optimizer.step()
        #####################################
        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        losses.update(loss.cpu().data.numpy())

        train_bar.set_description(
            'Train Epoch: [{}/{}], '
            'lr: {:.6f}, '
            'Loss: {:.3f}, '
            .format(epoch, args.epochs,
                    optimizer.param_groups[0]['lr'],
                    losses.avg, ))

        if i_debug // args.accumulation_steps >= args.training_step:
            break

    return {
        'loss': losses.avg,
        'mae': 0.0,
        'acc': 0.0,
        'c_index': 0.0,
        'metrics_': None,
        'metrics': None
    }


def direct_validate(model, valid_loader, criterion, args):
    """

    """
    model.eval()
    total_loss, total_num = 0.0, 0
    losses = AverageMeter()
    mae = AverageMeter()
    acc = AverageMeter()
    all_risk_probabilities = []
    all_followups = []
    all_risk_label = []

    with torch.no_grad():
        valid_bar = tqdm(valid_loader)
        i_debug = 0
        for input in valid_bar:
            if i_debug > 30 and args.debug:
                break
            i_debug += 1

            output = input_and_output(args, model, input)
            loss = compute_losses(args, criterion, input, output)

            risk_label = input['years_to_cancer'].cuda()
            years_last_followup = input['years_to_last_followup'].cuda()
            risk = output['risk']

            result = cal_mae_acc(risk, risk_label, years_last_followup, weights=args.time_to_events_weights)

            losses.update(loss.cpu().data.numpy())
            if result['count'] > 0:
                mae.update(result['mae'], n=result['count'])
                acc.update(result['acc'], n=result['count'])

            total_num += valid_loader.batch_size
            total_loss += loss.item() * valid_loader.batch_size

            valid_bar.set_description('Valid MAE: {:.4f}, Valid ACC: {:.2f}%, Valid Loss: {:.4f}, '
                    .format(mae.avg, acc.avg * 100, losses.avg, ))

            pred_risk_label = F.softmax(risk, dim=-1)

            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            all_risk_label.append(risk_label.cpu().numpy())
            all_followups.append(years_last_followup.cpu().numpy())

            if i_debug > args.val_step:
                break

        del input
        gc.collect()

    del valid_bar
    gc.collect()

    all_risk_probabilities = np.concatenate(all_risk_probabilities).reshape(-1, args.num_output_neurons)
    all_risk_label = np.concatenate(all_risk_label)
    all_followups = np.concatenate(all_followups)

    metrics_, _ = compute_auc_cindex(all_risk_probabilities, all_risk_label, all_followups, args.num_output_neurons,
                                    args.max_followup)

    try:
        logging.info('c-index is {:.4f} '.format(metrics_['c_index']))
    except:
        logging.info('c-index is None')

    try:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is {:.4f} '
                         .format(x, metrics_[x]))
    except:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is None')

    logging.info('ValLos:{:.2f}, '
                 'mae:{:.2f}, '
                 'acc:{:.2f}%, '
        .format(losses.avg,
        mae.avg, acc.avg * 100,
    ))

    return {
        'loss': losses.avg,
        'mae': mae.avg,
        'acc': acc.avg,
        'c_index': metrics_['c_index'],
        'metrics_': metrics_,
        'metrics': _
    }


def direct_test(model, test_loader, criterion, args, save_pkl=None, **kwargs):
    """

    """

    model.eval()

    total_loss, total_num = 0.0, 0
    losses = AverageMeter()
    mae = AverageMeter()
    acc = AverageMeter()
    all_risk_probabilities = []
    all_followups = []
    all_risk_label = []

    all_patient_ids = []
    all_exam_ids = []
    all_views = []
    all_lateralitys = []

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        i_debug = 0
        for input in test_bar:
            if i_debug > 30 and args.debug:
                break
            i_debug += 1

            output = input_and_output(args, model, input, inference=True)
            loss = compute_losses(args, criterion, input, output)

            risk_label = input['years_to_cancer'].cuda()
            years_last_followup = input['years_to_last_followup'].cuda()
            risk= output['risk']

            result = cal_mae_acc(risk, risk_label, years_last_followup, weights=args.time_to_events_weights)

            losses.update(loss.cpu().data.numpy())
            if result['count'] > 0:
                mae.update(result['mae'], n=result['count'])
                acc.update(result['acc'], n=result['count'])

            total_num += test_loader.batch_size
            total_loss += loss.item() * test_loader.batch_size

            test_bar.set_description('Test MAE: {:.4f}, Valid ACC: {:.2f}%, Valid Loss: {:.4f}, '
                                      .format(mae.avg, acc.avg*100, losses.avg, ))

            pred_risk_label = F.softmax(risk, dim=-1)

            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            all_risk_label.append(risk_label.cpu().numpy())
            all_followups.append(years_last_followup.cpu().numpy())

            all_patient_ids.append(input['pid'])
            all_exam_ids.append(input['exam_id'])
            all_views.append(input['view'])
            all_lateralitys.append(input['laterality'])

            if "test_step" in args and i_debug > args.test_step:
                break

        del input
        gc.collect()

    del test_bar
    gc.collect()

    all_patient_ids = np.concatenate(all_patient_ids)
    all_exam_ids = np.concatenate(all_exam_ids)
    all_views = np.concatenate(all_views)
    all_lateralitys = np.concatenate(all_lateralitys)
    all_risk_probabilities = np.concatenate(all_risk_probabilities).reshape(-1, args.num_output_neurons)
    all_risk_label = np.concatenate(all_risk_label)
    all_followups = np.concatenate(all_followups)

    metrics_, metrics = compute_auc_cindex(
        all_risk_probabilities, all_risk_label, all_followups, args.num_output_neurons,
        args.max_followup, confidence_interval=False)
    try:
        logging.info('c-index is {:.4f} '.format(metrics_['c_index']))
    except:
        logging.info('c-index is None')

    try:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is {:.4f} '
                         .format(x, metrics_[x]))
    except:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is None')

    logging.info('TestLos:{:.2f}, mae:{:.2f}, acc:{:.2f}%, '.format(losses.avg, mae.avg, acc.avg * 100,))

    if save_pkl is not None:
        save_dict = {
            'patient_id': all_patient_ids,
            'exam_id': all_exam_ids,
            'view': all_views,
            'laterality': all_lateralitys,
            'risk_probabilitie': all_risk_probabilities,
            'risk_label': all_risk_label,
            'followup': all_followups,
        }

        pickle.dump(save_dict, open('{}/result_{}.pkl'.format(args.results_dir, save_pkl), 'wb'))


    return {
        'loss': losses.avg,
        'mae': mae.avg,
        'acc': acc.avg,
        'c_index': metrics_['c_index'],
        'metrics_': metrics_,
        # 'metrics': metrics
    }


def get_train_val_test_demo(): #
    return direct_train, direct_validate, direct_test
