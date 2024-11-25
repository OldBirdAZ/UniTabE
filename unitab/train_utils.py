# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import os
import torch
import random
import numpy as np
from torch.nn import functional as F


from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_scheduler(
        name,
        optimizer,
        num_warmup_steps=None,
        num_training_steps=None,
):
    '''
    Unified API to get any scheduler from its name.

    Parameters
    ----------
    name: str
        The name of the scheduler to use.

    optimizer: torch.optim.Optimizer
        The optimizer that will be used during training.

    num_warmup_steps: int
        The number of warmup steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.

    num_training_steps: int
        The number of training steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    '''
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == 'constant':
        return schedule_func(optimizer)

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def self_supervised_contrastive_loss(features, temperature=10, base_temperature=10):
    """Compute the self-supervised VPCL loss.

    Parameters
    ----------
    :param features: torch.Tensor
        the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.
    :param temperature set as 10 in TransTab
    :param base_temperature set as 10 in TransTab
    Returns
    -------
    loss: torch.Tensor
        the computed self-supervised VPCL loss.
    """
    batch_size = features.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long, device=features.device).view(-1,1)
    mask = torch.eq(labels, labels.T).float().to(labels.device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 计算当前block相对于其他block的相似度最大值
    logits = anchor_dot_contrast - logits_max.detach()
    mask = mask.repeat(anchor_count, contrast_count)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    mid_log_val = -0.6931471824645996  # 0.5's log
    mean_acc = (mean_log_prob_pos > mid_log_val).float().mean()
    acc = mean_acc.mean().item()
    return loss, acc


def self_supervised_contrastive_loss_v2(neg_pair_ids, p1_vector, p2_vector):
    """
    :param neg_pair_ids: LongTensor, [N]
    :param p1_vector: Tensor, [N, hidden_size]
    :param p2_vector: Tensor, [N, hidden_size]
    :return:
    """
    neg_select_index = neg_pair_ids.unsqueeze(1).repeat(1, p1_vector.size(1))
    p3_vector = p1_vector.gather(0, neg_select_index)
    # [N, 1, 2]
    logits = torch.matmul(p1_vector.unsqueeze(1), torch.cat((p2_vector.unsqueeze(2), p3_vector.unsqueeze(2)), 2))
    logits = logits.squeeze(1)
    loss = torch.mean(-F.log_softmax(logits, 1)[:, 0])
    acc = torch.mean((logits[:, 0] > logits[:, 1]).float()).item()
    return loss, acc

