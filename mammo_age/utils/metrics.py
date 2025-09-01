import torch
import numpy as np
import torch.nn.functional as F


# def cal_mae_acc_rank(logits, targets, is_sto=True):
#     if is_sto:
#         r_dim, s_dim, out_dim = logits.shape
#         assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
#         logits = logits.view(r_dim, s_dim, out_dim / 2, 2)
#         logits = torch.argmax(logits, dim=-1)
#         logits = torch.sum(logits, dim=-1)
#         logits = torch.mean(logits.float(), dim=0)
#         logits = logits.cpu().data.numpy()
#         targets = targets.cpu().data.numpy()
#         mae = sum(abs(logits - targets)) * 1.0 / len(targets)
#         acc = sum(np.rint(logits) == targets) * 0.01 / len(targets)
#     else:
#         s_dim, out_dim = logits.shape
#         assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
#         logits = logits.view(s_dim, out_dim / 2, 2)
#         logits = torch.argmax(logits, dim=-1)
#         logits = torch.sum(logits, dim=-1)
#         logits = logits.cpu().data.numpy()
#         targets = targets.cpu().data.numpy()
#         mae = sum(abs(logits - targets)) * 1.0 / len(targets)
#         acc = sum(np.rint(logits) == np.rint(targets)) * 0.01 / len(targets)
#     return mae, acc


def cal_mae_acc_reg(logits, targets, is_sto=False, threshold=5):
    if is_sto:
        logits = logits.mean(dim=0)

    assert logits.view(-1).shape == targets.shape, "logits {}, targets {}".format(
        logits.shape, targets.shape)

    logits = logits.cpu().data.numpy().reshape(-1)
    targets = targets.cpu().data.numpy()
    # mae_batch = abs(logits - targets) * 100.0
    error_batch = logits - targets
    mae_batch = abs(error_batch)
    mae = sum(mae_batch) / len(targets)
    # acc = sum(np.rint(abs(logits - targets) * 100.0) <= threshold) * 1.0 / len(targets)
    acc = sum(np.rint(abs(logits - targets)) <= threshold) * 1.0 / len(targets)

    return {
        'mae': mae,
        'acc': acc,
        'mae_batch': mae_batch,
        'error_batch': error_batch,
        'pred_age': logits,
            }


def cal_mae_acc_cls(logits, targets, is_sto=False, threshold=5, GL_multiview=False):
    if GL_multiview == True:
        if is_sto:
            r_dim, s_dim, out_dim = logits[0].shape
        else:
            s_dim, out_dim = logits[0].shape

        exp_datas = []
        exp_data = np.zeros(s_dim)
        for index_ in range(len(logits)):
            if is_sto:
                # label_arr = torch.arange(0, out_dim).float().cuda()
                probs = F.softmax(logits[index_], -1)
                # probs_data = probs.cpu().data.numpy()
                # label_arr = np.array(range(out_dim))
                label_arr = torch.arange(0, out_dim).float().cuda()
                exp = torch.sum(probs * label_arr, dim=-1)
                exp = torch.mean(exp, dim=0)
                exp_data += exp.cpu().data.numpy()
                exp_datas.append(exp.cpu().data.numpy())
                # max_a = torch.mean(probs, dim=0)
                # max_data = max_a.cpu().data.numpy()
                # max_data = np.argmax(max_data, axis=1)
                # exp_data = exp.cpu().data.numpy()
                # exp_data += np.sum(exp_data, axis=1)
            else:
                # s_dim, out_dim = logits[index_].shape
                probs = F.softmax(logits[index_], -1)
                probs_data = probs.cpu().data.numpy()
                # max_data = np.argmax(probs_data, axis=1)
                # label_arr = np.array(range(out_dim)) + 1
                # label_arr = np.array(range(out_dim)) + 22
                label_arr = np.array(range(out_dim))
                # print(label_arr)
                exp_data += np.sum(probs_data * label_arr, axis=1)
                exp_datas.append(np.sum(probs_data * label_arr, axis=1))
        exp_data = exp_data / len(logits)
    else:
        exp_datas = None
        if is_sto:
            r_dim, s_dim, out_dim = logits.shape
            label_arr = torch.arange(0, out_dim).float().cuda()
            probs = F.softmax(logits, -1)
            exp = torch.sum(probs * label_arr, dim=-1)
            exp = torch.mean(exp, dim=0)
            max_a = torch.mean(probs, dim=0)
            max_data = max_a.cpu().data.numpy()
            max_data = np.argmax(max_data, axis=1)
            exp_data = exp.cpu().data.numpy()
        else:
            s_dim, out_dim = logits.shape
            # print(s_dim, out_dim)
            probs = F.softmax(logits, -1)
            probs_data = probs.cpu().data.numpy()
            max_data = np.argmax(probs_data, axis=1)
            # label_arr = np.array(range(out_dim)) + 1
            # label_arr = np.array(range(out_dim)) + 22
            label_arr = np.array(range(out_dim))
            # print(label_arr)
            exp_data = np.sum(probs_data * label_arr, axis=1)
    target_data = targets.cpu().data.numpy()
    # print('exp_data', exp_data)
    # mae_batch = abs(exp_data - (target_data * 100)) * 1.0
    error_batch = exp_data - target_data
    mae_batch = abs(error_batch) * 1.0
    mae = sum(mae_batch) / len(target_data)
    # acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)
    # acc = sum(np.rint(abs(exp_data - (target_data * 100))) <= threshold) * 1.0 / len(target_data)
    acc = sum(np.rint(abs(exp_data - target_data)) <= threshold) * 1.0 / len(target_data)
    result_dict = {'mae': mae, 'acc': acc, 'mae_batch': mae_batch, 'error_batch': error_batch, 'pred_age': exp_data,}

    if exp_datas is not None:
        for i in range(len(exp_datas)):
            result_dict['pred_age_{}'.format(i)] = exp_datas[i]
            result_dict['pred_age_max'.format(i)] = np.max(exp_datas, axis=0)
            result_dict['pred_age_min'.format(i)] = np.min(exp_datas, axis=0)

    return result_dict


def get_metric(main_loss_type, **kwargs):
    assert main_loss_type in ['cls', 'reg', 'rank'], \
        "main_loss_type not in ['cls', 'reg', 'rank'], loss type {%s}" % (
            main_loss_type)
    if main_loss_type == 'cls':
        return cal_mae_acc_cls
    elif main_loss_type == 'reg':
        return cal_mae_acc_reg
    else:
        raise AttributeError('main loss type: {}'.format(main_loss_type))
