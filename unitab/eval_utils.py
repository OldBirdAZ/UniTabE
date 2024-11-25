# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def evaluate(y_pred, y_target, metric='auc', seed=123):
    """
    :param y_pred: list, [N, ] or [N, num_classes]
    :param y_target: list, [N, ]
    :param metric:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    y_pred = np.array(y_pred)
    y_target = np.array(y_target)
    eval_fn = get_eval_metric_fn(metric)
    auc_list = []
    stats_dict = defaultdict(list)
    for i in range(10):
        sub_idx = np.random.choice(np.arange(len(y_pred)), len(y_pred), replace=True)
        sub_ypred = y_pred[sub_idx]
        sub_ytest = y_target[sub_idx]
        try:
            sub_res = eval_fn(sub_ytest, sub_ypred)
            stats_dict[metric].append(sub_res)
        except ValueError:
            print('evaluation went wrong!')
    for key in stats_dict.keys():
        stats = stats_dict[key]
        alpha = 0.95
        p = ((1-alpha)/2) * 100
        lower = max(0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('{} || alpha {:.2f}, mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
        if key == metric: auc_list.append((upper+lower)/2)
    return auc_list


def get_eval_metric_fn(eval_metric):
    fn_dict = {
        'acc': acc_fn,
        'auc': auc_fn,
        'mse': mse_fn,
        'val_loss': None,
    }
    return fn_dict[eval_metric]


def acc_fn(y, p):
    y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)


# def auc_fn(y, p):
#     return roc_auc_score(y, p)
def auc_fn(y, p):
    # multi_class_mode = False
    if len(p.shape) > 1:
        # multi_class_mode = True
        return roc_auc_score(y, p, multi_class='ovo')
    return roc_auc_score(y, p)


def mse_fn(y, p):
    return mean_squared_error(y, p)
