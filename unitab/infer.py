# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.model_builder import build_pretrain_model
from unitab.model_builder import build_data_collator
from unitab.tab_data_process import TokenizerProxy
from unitab.tab_data_loader import build_data_loader
import os
import torch
from unitab.eval_utils import evaluate
from unitab import common_const
import json


class TabInfer:
    def __init__(self, config):
        self.config = config
        config.test_mode = True
        self.config.test_mode = True
        self.num_classification_labels = config.num_classification_labels
        self.test_out_path = config.test_out_path
        self.test_max_dec_steps = config.test_max_dec_steps
        self.test_single_id2tok = config.test_single_id2tok

        self.tokenizer_proxy = TokenizerProxy(max_tok_len=config.max_tok_len)
        pad_id = self.tokenizer_proxy.pad_token_id
        vocab_size = self.tokenizer_proxy.vocab_size
        self.missing_value_token = self.tokenizer_proxy.mask_token
        model = build_pretrain_model(config, pad_id, vocab_size)
        if config.restore_path is not None:
            print('Restoring Parameters From: {}'.format(config.restore_path))
            state_dict = torch.load(config.restore_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        self.model = model.cuda()
        task_names = config.task_names
        task_alphas = config.task_alphas
        self.pretrain_task_names = [tn for tn in task_names.split(',') if len(tn) > 1][:1]
        print('Pre-Training Tasks: {}'.format(self.pretrain_task_names))
        task_alphas = [float(tn) for tn in task_alphas.split(',')][:1]
        assert len(task_alphas) == len(self.pretrain_task_names)
        self.task_alphas = {k: v for k,v in zip(self.pretrain_task_names, task_alphas)}
        print('task_alphas: {}'.format(self.task_alphas))
        self.batch_data_collate_fn = build_data_collator(config, self.tokenizer_proxy, self.pretrain_task_names)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.task_infer_logs = {tn: None for tn in self.pretrain_task_names}
        test_data_path = config.test_data_path
        assert test_data_path is not None and len(test_data_path) > 1
        self.test_loader = self.build_data_loader(test_data_path, shuffle=False, drop_last=False, single_worker=True)

    def is_classification_tasks(self):
        if 1 == len(self.pretrain_task_names) and 1 == len(self.task_alphas):
            if self.pretrain_task_names[0] in [
                common_const.TASK_NAME_CLASSIFICATION,
            ]:
                return True
        return False

    def is_common_predict_tasks(self):
        if 1 == len(self.pretrain_task_names) and 1 == len(self.task_alphas):
            if self.pretrain_task_names[0] in [
                common_const.TASK_NAME_COMMON_PREDICTION,
                common_const.TASK_NAME_COMMON_PREDICTION_SEP_ANSWER,
                common_const.TASK_NAME_COMMON_FILL_MISSVAL_PREDICTION,
            ]:
                return True
        return False

    def build_data_loader(self, data_path, shuffle=True, drop_last=True, single_worker=False):
        loader = build_data_loader(
            data_path,
            collate_fn=self.batch_data_collate_fn,
            missing_value_token=self.missing_value_token,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers if not single_worker else 1,
            drop_last=drop_last,
            retrieve_label_outof_fs=self.config.retrieve_label_outof_fs,
            label_column_name=self.config.label_column_name,
            reader_num_processes=self.config.reader_num_processes,
            num_precision=self.config.num_precision,
            target_label_mapping=self.config.target_label_mapping
        )
        return loader

    def put_data_to_gpu(self, wrapped_data):
        if isinstance(wrapped_data, dict):
            for k, v in wrapped_data.items():
                if torch.is_tensor(v):
                    wrapped_data[k] = v.cuda()
            if 'basic_data' in wrapped_data:
                for k, v in wrapped_data['basic_data'].items():
                    if torch.is_tensor(v):
                        wrapped_data['basic_data'][k] = v.cuda()
        elif isinstance(wrapped_data, list):
            for wd in wrapped_data:
                for k, v in wd.items():
                    if torch.is_tensor(v):
                        wd[k] = v.cuda()
                if 'basic_data' in wd:
                    for k, v in wd['basic_data'].items():
                        if torch.is_tensor(v):
                            wd['basic_data'][k] = v.cuda()

    def reset_logs(self, logs):
        for tn in logs.keys():
            cur_log = logs[tn]
            for kk in cur_log.keys():
                cur_log[kk] = 0.0

    def do_infer(self):
        if self.config.test_export_representation:
            self.export_representation_infer()
        elif self.is_classification_tasks():
            self.classification_infer()
        else:
            self.common_infer()
        # self.common_infer()

    def classification_infer(self):
        print('validate ...')
        save_file = open(self.test_out_path, 'wt', encoding='utf-8')
        self.model.eval()
        old_dec_temperature = self.model.dec_temperature
        self.model.dec_temperature = 1.0
        metrics_record = {}
        avg_auc = []
        for batch_wrapped_data in self.test_loader:
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.model(cur_task_data, task)
                log_vals = result['log_vals']
                if self.task_infer_logs[task] is None:
                    self.task_infer_logs[task] = log_vals
                    log_vals['step'] = 1.0
                else:
                    self.task_infer_logs[task]['step'] += 1.0
                    for k, v in log_vals.items():
                        self.task_infer_logs[task][k] += v
                if 'classification_probs' in result and result['classification_probs'] is not None:
                    if task not in metrics_record:
                        metrics_record[task] = {
                            'y_prob_tgt': [],
                            'y_pred': [],
                            'y_label': []
                        }
                    batch_pred_probs = result['classification_probs'].tolist()
                    batch_labels = cur_task_data['batch_target_label'].tolist()
                    batch_pred_labels = torch.argmax(result['classification_probs'], 1).tolist()
                    bidx = -1
                    batch_raw_data = None
                    if 'batch_raw_data' in cur_task_data:
                        batch_raw_data = cur_task_data['batch_raw_data']
                    batch_raw_target = None
                    if 'batch_raw_target' in cur_task_data:
                        batch_raw_target = cur_task_data['batch_raw_target']
                    for cur_pred_probs, cur_pred_label, cur_label in zip(batch_pred_probs, batch_pred_labels, batch_labels):
                        bidx += 1
                        cur_raw_data = None
                        if batch_raw_data is not None:
                            cur_raw_data = json.loads(batch_raw_data[bidx])
                        cur_raw_target = None
                        if batch_raw_target is not None:
                            cur_raw_target = batch_raw_target[bidx]
                        save_item = {
                            'raw_data': cur_raw_data,
                            'raw_target': cur_raw_target,
                            'pred_probs': cur_pred_probs,
                            'pred_result': cur_pred_label,
                            'ground_truth': cur_label
                        }
                        save_file.write('{}\n'.format(json.dumps(save_item)))
                    save_file.flush()
                    # pred_prob_of_tgt = result['classification_probs'].gather(1, cur_task_data['batch_target_label'].unsqueeze(1)).squeeze(1).tolist()
                    # 临时针对 2分类, 多分类在计算调用 roc_auc_score(Y_test, Y_pred_prob, multi_class='ovo'), Y_pred_prob shape [N, num_classes]
                    if 2 == self.num_classification_labels:
                        pred_prob_of_tgt = result['classification_probs'][:, 1].tolist()
                    else:
                        pred_prob_of_tgt = result['classification_probs'].tolist()
                    metrics_record[task]['y_prob_tgt'] += pred_prob_of_tgt
                    metrics_record[task]['y_pred'] += result['classification_probs'].tolist()
                    metrics_record[task]['y_label'] += cur_task_data['batch_target_label'].tolist()
        for task in self.pretrain_task_names:
            cur_log_step = 1 if self.task_infer_logs[task]['step'] == 0 else self.task_infer_logs[task]['step']
            cur_log_str = ','.join(
                [' {}:{}'.format(k, round(v / cur_log_step, 6)) for k, v in self.task_infer_logs[task].items() if k != 'step']
            )
            print('## {}|| {}'.format(task, cur_log_str))
            if task in metrics_record:
                task_auc = evaluate(metrics_record[task]['y_prob_tgt'], metrics_record[task]['y_label'], metric='auc', seed=123)
                task_acc = evaluate(metrics_record[task]['y_pred'], metrics_record[task]['y_label'], metric='acc', seed=123)
                avg_auc += task_auc
        self.reset_logs(self.task_infer_logs)
        self.model.dec_temperature = old_dec_temperature
        if len(avg_auc) < 1:
            return None
        return sum(avg_auc) / len(avg_auc)

    def common_infer(self):
        print('common validate ...')
        save_file = open(self.test_out_path, 'wt', encoding='utf-8')
        self.model.eval()
        for batch_wrapped_data in self.test_loader:
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.model.infer_common(cur_task_data, max_dec_steps=self.test_max_dec_steps)
                dec_out_ids = result['dec_out_ids']
                batch_size = result['batch_size']
                batch_gen_infoned_probs = None
                if 'gen_infoned_probs' in result and result['gen_infoned_probs'] is not None:
                    batch_gen_infoned_probs = result['gen_infoned_probs'].tolist()
                if self.test_single_id2tok:
                    batch_dec_texts = dec_ids_to_str_single_tok(dec_out_ids, self.tokenizer_proxy)
                else:
                    batch_dec_texts = dec_ids_to_str(dec_out_ids, self.tokenizer_proxy)
                if self.test_single_id2tok:
                    batch_ground_texts = dec_ids_to_str_single_tok(cur_task_data['batch_value_eos_ids'], self.tokenizer_proxy)
                else:
                    batch_ground_texts = dec_ids_to_str(cur_task_data['batch_value_eos_ids'], self.tokenizer_proxy)
                # print('batch_dec_texts: {}'.format(batch_dec_texts))
                # print('batch_raw_dec_texts: {}'.format(batch_raw_dec_texts))
                # exit(0)
                bidx = -1
                batch_raw_data = None
                if 'batch_raw_data' in cur_task_data:
                    batch_raw_data = cur_task_data['batch_raw_data']
                batch_raw_target = None
                if 'batch_raw_target' in cur_task_data:
                    batch_raw_target = cur_task_data['batch_raw_target']
                for dec_text in batch_dec_texts:
                    bidx += 1
                    cur_raw_data = None
                    if batch_raw_data is not None:
                        cur_raw_data = json.loads(batch_raw_data[bidx])
                    cur_raw_target = None
                    if batch_raw_target is not None:
                        cur_raw_target = batch_raw_target[bidx]
                    cur_gen_infoned_probs = None
                    if batch_gen_infoned_probs is not None:
                        cur_gen_infoned_probs = batch_gen_infoned_probs[bidx]
                    save_file.write('{}\n'.format(json.dumps({
                        'raw_data': cur_raw_data,
                        'raw_target': cur_raw_target,
                        'pred_result': dec_text,
                        'gen_infoned_probs': cur_gen_infoned_probs,
                        'tok_target': batch_ground_texts[bidx],
                    })))
                save_file.flush()

    def export_representation_infer(self):
        print('exporting representation ...')
        save_file = open(self.test_out_path, 'wt', encoding='utf-8')
        self.model.eval()
        for batch_wrapped_data in self.test_loader:
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.model.export_representation(cur_task_data)
                cur_batch_states = result['states']
                cur_batch_final_mask = result['batch_final_mask']
                cur_batch_vector_rep = cur_batch_states[:, 0]
                cur_batch_avg_rep = torch.sum(
                    cur_batch_states.masked_fill(~cur_batch_final_mask.unsqueeze(2), 0.0),
                    1) / torch.sum(cur_batch_final_mask, 1).unsqueeze(1)
                batch_raw_data = None
                if 'batch_raw_data' in cur_task_data:
                    batch_raw_data = cur_task_data['batch_raw_data']
                cur_batch_size = cur_batch_vector_rep.size(0)
                cur_batch_vector_rep_val = cur_batch_vector_rep.tolist()
                cur_batch_avg_rep_val = cur_batch_avg_rep.tolist()
                for bidx in range(cur_batch_size):
                    cur_raw_data = None
                    if batch_raw_data is not None:
                        cur_raw_data = json.loads(batch_raw_data[bidx])
                    save_file.write('{}\n'.format(json.dumps({
                        'raw_data': cur_raw_data,
                        'export_representation': cur_batch_vector_rep_val[bidx],
                        'export_avg_representation': cur_batch_avg_rep_val[bidx],
                    })))
                save_file.flush()
        pass


# def dec_ids_to_str(dec_out_valid, tokenizer_proxy):
#     dec_ids = torch.argmax(dec_out_valid, 2)
#     dec_texts = []
#     for text_ids in dec_ids.tolist():
#         # print('text_ids: {}'.format(text_ids))
#         dec_texts.append(tokenizer_proxy.tokenizer.decode(text_ids))
#         # print('dec_text: {}'.format(dec_texts[-1]))
#     return dec_texts
def dec_ids_to_str(dec_out_ids, tokenizer_proxy):
    eos_tok_id = tokenizer_proxy.eos_token_id
    dec_ids = dec_out_ids.tolist()
    dec_texts = []
    for text_ids in dec_ids:
        if eos_tok_id in text_ids:
            eos_index = text_ids.index(eos_tok_id)
            if eos_index > 0:
                text_ids = text_ids[:eos_index]
        # print('text_ids: {}'.format(text_ids))
        dec_texts.append(tokenizer_proxy.tokenizer.decode(text_ids))
        # print('dec_text: {}'.format(dec_texts[-1]))
    return dec_texts


def dec_ids_to_str_single_tok(dec_out_ids, tokenizer_proxy):
    eos_tok_id = tokenizer_proxy.eos_token_id
    dec_ids = dec_out_ids.tolist()
    dec_texts = []
    for text_ids in dec_ids:
        if eos_tok_id in text_ids:
            eos_index = text_ids.index(eos_tok_id)
            if eos_index > 0:
                text_ids = text_ids[:eos_index]
        # print('text_ids: {}'.format(text_ids))
        cur_toks = [tokenizer_proxy.tokenizer.decode(tidx) for tidx in text_ids]
        dec_texts.append(' '.join(cur_toks))
        # print('dec_text: {}'.format(dec_texts[-1]))
    return dec_texts


