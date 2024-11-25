# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import argparse


def set_model_args(parser):
    parser.add_argument('-n_data_type', '--n_data_type', type=int, default=3)
    parser.add_argument('-emb_size', '--emb_size', type=int, default=512)
    parser.add_argument('-hidden_size', '--hidden_size', type=int, default=512)
    parser.add_argument('-n_head', '--n_head', type=int, default=8)
    parser.add_argument('-ffn_size', '--ffn_size', type=int, default=1024)
    parser.add_argument('-n_enc_layer', '--n_enc_layer', type=int, default=6)
    parser.add_argument('-n_dec_layer', '--n_dec_layer', type=int, default=1)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.1)
    parser.add_argument('-temperature', '--temperature', type=float, default=1.0)
    parser.add_argument('-datatype_aware', '--datatype_aware', action='store_true', default=False, help='use datatype')
    parser.add_argument('-use_memory_efficient', '--use_memory_efficient', action='store_true', default=False, help='gpu memory efficient, will sacrifice performance')


def set_inference_args(parser):
    parser.add_argument('-test_export_representation', '--test_export_representation', action='store_true', default=False, help='only save best ckpt')
    parser.add_argument('-test_export_states', '--test_export_states', action='store_true', default=False, help='only save best ckpt')
    parser.add_argument('-test_single_id2tok', '--test_single_id2tok', action='store_true', default=False, help='')


def set_data_args(parser):
    parser.add_argument('-max_tok_len', '--max_tok_len', type=int, default=100)
    parser.add_argument('-max_feature_tok_len', '--max_feature_tok_len', type=int, default=50)
    parser.add_argument('-max_prompt_tok_len', '--max_prompt_tok_len', type=int, default=100)
    parser.add_argument('-max_num_features', '--max_num_features', type=int, default=-1, help='-1 means using all')

    # contrastive learning config
    parser.add_argument('-num_precision', '--num_precision', type=int, default=4)
    parser.add_argument('-num_partition', '--num_partition', type=int, default=4)
    parser.add_argument('-overlap_ratio', '--overlap_ratio', type=float, default=0.5)
    parser.add_argument('-num_classification_labels', '--num_classification_labels', type=int, default=2)
    # data loader
    parser.add_argument('-retrieve_label_outof_fs', '--retrieve_label_outof_fs', action='store_true', default=False,
                        help='retrieve label column from features columns')
    parser.add_argument('-label_column_name', '--label_column_name', type=str, default='target_label', help='')
    parser.add_argument('-removed_features', '--removed_features', type=str, default='', help='remove some columns')
    parser.add_argument('-reader_num_processes', '--reader_num_processes', type=int, default=1)
    parser.add_argument('-test_mode', '--test_mode', type=bool, default=False)
    parser.add_argument('-test_max_dec_steps', '--test_max_dec_steps', type=int, default=1)
    parser.add_argument('-common_pred_prefix', '--common_pred_prefix', type=str, default='predict label :')
    parser.add_argument('-common_classification_prefix', '--common_classification_prefix', type=str, default='classification :')
    parser.add_argument('-test_data_path', '--test_data_path', type=str, default='')
    parser.add_argument('-test_out_path', '--test_out_path', type=str, default='./test_out.jsonl')
    parser.add_argument('-target_label_mapping', '--target_label_mapping', type=str, default='', help='map string target_label to int value, e.g. N:0|Y:1')
    parser.add_argument('-only_save_best', '--only_save_best', action='store_true', default=False, help='only save best ckpt')
    parser.add_argument('-no_save_raw', '--no_save_raw', action='store_true', default=False, help='')
    parser.add_argument('-dynamic_mask_missval_span', '--dynamic_mask_missval_span', type=str, default='1.0:1|1.25:2|1.5:3|1.75:4|2.0:5|2.5:7')
    parser.add_argument('-dynamic_mask_span_max_th', '--dynamic_mask_span_max_th', type=float, default=0.5)
    parser.add_argument('-common_fill_missval_column', '--common_fill_missval_column', type=str, default=None)
    parser.add_argument('-gen_constrained_toks', '--gen_constrained_toks', type=str, default=None, help='constrain the generation within mini vocab range')
    parser.add_argument('-gen_constrained_sep', '--gen_constrained_sep', type=str, default=',')


def set_train_args():
    parser = argparse.ArgumentParser(description="Tabular Pre-training Config")
    # model args
    set_model_args(parser)
    # set data args
    set_data_args(parser)
    # train args
    parser.add_argument('-save_dir', '--save_dir', type=str, default='../outputs/run1')
    parser.add_argument('-restore_path', '--restore_path', type=str, default=None)
    parser.add_argument('-strict_restore', '--strict_restore', action='store_true', default=False)
    parser.add_argument('-train_data_dir', '--train_data_dir', type=str, required=False)
    parser.add_argument('-valid_data_path', '--valid_data_path', type=str, default=None, required=False)
    parser.add_argument('-train_mode', '--train_mode', type=str, default='default', choices=['default', 'sep_each_task', 'default_bp_each_task'])
    # use ',' to separate different tasks;
    #    also need to specific the loss weight for each task by setting the param task_alpha
    parser.add_argument('-task_names', '--task_names', type=str, default='scl_task', help='multiple tasks separated by,', required=True)
    parser.add_argument('-task_alphas', '--task_alphas', type=str, default='1.0', help='multiple tasks separated by,', required=True)
    parser.add_argument('-num_epoch', '--num_epoch', type=int, default=50)
    parser.add_argument('-freeze_encoder_steps', '--freeze_encoder_steps', type=int, default=0)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64)
    parser.add_argument('-valid_batch_size', '--valid_batch_size', type=int, default=None)
    parser.add_argument('-random_seed', '--random_seed', type=int, default=42)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-use_grad_clip', '--use_grad_clip', action='store_true', default=False)
    parser.add_argument('-max_grad_clip', '--max_grad_clip', type=float, default=1.0)
    parser.add_argument('-num_workers', '--num_workers', type=int, default=4)
    parser.add_argument('-log_interval', '--log_interval', type=int, default=100)
    parser.add_argument('-save_ckpt_interval', '--save_ckpt_interval', type=int, default=10000)
    parser.add_argument('-n_g_accum', '--n_g_accum', type=int, default=10)
    # parser.add_argument('-single_task_data_mode', '--single_task_data_mode', action='store_true', default=False)
    parser.add_argument('-ft_reset_decoder', '--ft_reset_decoder', action='store_true', default=False, help='finetune reset dec params')
    parser.add_argument('-optm_name', '--optm_name', type=str, default='Adam', help='')
    parser.add_argument('-use_f16', '--use_f16', action='store_true', default=False, help='using half precision training, in this implementation, it saves around 1/6 gpu memory')

    parser.add_argument('-multi_gpu_ids', '--multi_gpu_ids', type=str, default='0,1,2,3')

    parser.add_argument('-valid_metric', '--valid_metric', type=str, default='valid_auc', help='acc, loss, valid_acc, valid_auc')
    parser.add_argument('-valid_smaller_better', '--valid_smaller_better', action='store_true', default=False, help='acc, loss')
    parser.add_argument('-early_stop', '--early_stop', action='store_true', default=False, help='')
    parser.add_argument('-early_stop_patience', '--early_stop_patience', type=int, default=10)
    parser.add_argument('-trunc_loading_max_num_examples', '--trunc_loading_max_num_examples', type=int, default=-1, help='trunc dataset to max num size')
    parser.add_argument('-freeze_enc_layers', '--freeze_enc_layers', type=str, default=None, help='layer number, sep with comma')
    parser.add_argument('-ft_freeze_encoder', '--ft_freeze_encoder', action='store_true', default=False, help='finetune reset dec params')
    # imbalance classification
    parser.add_argument('-use_cls_focal', '--use_cls_focal', action='store_true', default=False, help='whether using focal loss')
    parser.add_argument('-focal_gamma', '--focal_gamma', type=float, default=0.0, help='bigger-->focus more on hard examples')

    # inference settings
    set_inference_args(parser)

    args = parser.parse_args()

    return args


def set_infer_cli_args():
    parser = argparse.ArgumentParser(description="Tabular Pre-training Config")
    parser.add_argument('-model_path', '--model_path', type=str, required=True)
    parser.add_argument('-data_meta_path', '--data_meta_path', type=str, required=True)
    parser.add_argument('-train_config_path', '--train_config_path', type=str, required=True)
    parser.add_argument('-no_gpu', '--no_gpu', action='store_true', default=False)
    parser.add_argument('-task_name', '--task_name', type=str, default='classification', help='common_predict, classification')
    # parser.add_argument('-example_str', '--example_str', type=str, required=True, help='different column separated by ``[COLSEP]``')
    parser.add_argument('-test_max_dec_steps', '--test_max_dec_steps', type=int, default=1)

    args = parser.parse_args()
    return args



