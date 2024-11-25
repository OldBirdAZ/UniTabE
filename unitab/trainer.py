# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.model_builder import build_pretrain_model
from unitab.model_builder import build_data_collator
from unitab.tab_data_process import TokenizerProxy
from unitab.tab_data_loader import build_data_loader
import os
import random
import torch
from torch import nn
from unitab.train_utils import get_parameter_names, random_seed
from unitab.eval_utils import evaluate
from unitab import common_const
from unitab.extra_optimizators import Lion
from unitab.loss_utils import FocalLoss


class TabTrainer:
    def __init__(self, config):
        self.config = config
        random_seed(config.random_seed)
        self.tokenizer_proxy = TokenizerProxy(max_tok_len=config.max_tok_len)
        save_dir = config.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.tokenizer_proxy.save_tokenizer_to_disk(save_dir)
        print('## saving config')
        torch.save(config, os.path.join(save_dir, 'train_config.pt'))

        pad_id = self.tokenizer_proxy.pad_token_id
        vocab_size = self.tokenizer_proxy.vocab_size
        self.missing_value_token = self.tokenizer_proxy.mask_token
        model = build_pretrain_model(config, pad_id, vocab_size)
        self.try_to_inject_classification_focal_loss(model)
        total_params = sum([param.nelement() for param in model.parameters()])
        print('## total_params: {}'.format(total_params))
        total_enc_params = sum([param.nelement() for param in model.tab_encoder.parameters()])
        print('## total encoder params: {}'.format(total_enc_params))
        old_dec_state = model.tab_decoder.state_dict()
        if config.restore_path is not None:
            print('Restoring Parameters From: {}'.format(config.restore_path))
            state_dict = torch.load(config.restore_path, map_location='cpu')
            # safe checking...
            if config.ft_reset_decoder:
                from collections import OrderedDict
                ft_state_dict = OrderedDict()
                for pn in state_dict.keys():
                    if pn.startswith('tab_decoder') and not pn.startswith('tab_decoder.word_embed'):
                        # print('delete {}'.format(pn))
                        continue
                    ft_state_dict[pn] = state_dict[pn]
                state_dict = ft_state_dict
            model.load_state_dict(state_dict, strict=config.strict_restore)
        if config.ft_reset_decoder:
            print('Resetting Dec Parameters ... ')
            new_dec_state = model.tab_decoder.state_dict()
            old_dec_state['word_embed.weight'] = new_dec_state['word_embed.weight']
            model.tab_decoder.load_state_dict(old_dec_state)
        self.model = model.cuda()
        task_names = config.task_names
        task_alphas = config.task_alphas
        self.pretrain_task_names = [tn for tn in task_names.split(',') if len(tn) > 1]
        print('Pre-Training Tasks: {}'.format(self.pretrain_task_names))
        task_alphas = [float(tn) for tn in task_alphas.split(',')]
        assert len(task_alphas) == len(self.pretrain_task_names)
        self.task_alphas = {k: v for k,v in zip(self.pretrain_task_names, task_alphas)}
        print('task_alphas: {}'.format(self.task_alphas))
        self.batch_data_collate_fn = build_data_collator(config, self.tokenizer_proxy, self.pretrain_task_names)
        train_data_dir = config.train_data_dir
        files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
        train_files = [f for f in files if os.path.isfile(f) and f.endswith('.jsonl')]
        self.train_files = train_files
        self.num_epoch = config.num_epoch
        self.batch_size = config.batch_size
        self.valid_batch_size = config.valid_batch_size if config.valid_batch_size is not None else self.batch_size
        self.num_workers = config.num_workers
        self.task_train_logs = {tn: None for tn in self.pretrain_task_names}
        self.task_valid_logs = {tn: None for tn in self.pretrain_task_names}
        self.optimizer = None
        self.optm_fn = torch.optim.Adam
        # torch.optim.RAdam
        if config.optm_name == 'Lion':
            self.optm_fn = Lion
        else:
            self.optm_fn = getattr(torch.optim, config.optm_name)
        self._create_optimizer(config.weight_decay, config.lr)
        self._create_dec_optimizer(config.weight_decay, config.lr)
        self.n_g_accum = config.n_g_accum
        valid_data_path = config.valid_data_path
        self.valid_loader = None
        if valid_data_path is not None:
            self.valid_loader = self.build_data_loader(valid_data_path, self.valid_batch_size, shuffle=False, drop_last=False, single_worker=True)
        self.freeze_encoder_steps = config.freeze_encoder_steps
        self.save_ckpt_interval = config.save_ckpt_interval
        self.valid_metric = config.valid_metric
        self.valid_smaller_better = config.valid_smaller_better
        self.early_stop = config.early_stop
        self.early_stop_patience = config.early_stop_patience
        self.num_classification_labels = config.num_classification_labels
        self.only_save_best = config.only_save_best
        self.no_save_raw = config.no_save_raw

    def is_classification_tasks(self):
        if 1 == len(self.pretrain_task_names) and 1 == len(self.task_alphas):
            if self.pretrain_task_names[0] in [
                common_const.TASK_NAME_CLASSIFICATION,
            ]:
                return True
        return False

    def is_common_predict_tasks(self):
        if 1 == len(self.pretrain_task_names) and 1 == len(self.task_alphas):
            if self.pretrain_task_names[0] not in [
                common_const.TASK_NAME_CLASSIFICATION,
            ]:
                return True
        return False

    def build_data_loader(self, data_path, batch_size, shuffle=True, drop_last=True, single_worker=False):
        loader = build_data_loader(
            data_path,
            collate_fn=self.batch_data_collate_fn,
            missing_value_token=self.missing_value_token,
            batch_size=batch_size,
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

    def _create_optimizer(self, weight_decay, lr):
        if self.optimizer is None:
            optim_params = self.model.parameters()
            if self.config.ft_freeze_encoder:
                optim_params = []
                for name, tensor in self.model.tab_decoder.named_parameters():
                    if 'word_embed' not in name:
                        optim_params.append(tensor)
            else:
                if self.config.freeze_enc_layers is not None and len(self.config.freeze_enc_layers) > 0:
                    enc_layer_prefix = "tab_encoder.encoding_block.layers.{}"
                    freeze_enc_layers = [enc_layer_prefix.format(lidx) for lidx in self.config.freeze_enc_layers.split(',') if len(lidx.strip()) > 0]
                    optim_params = []
                    frozen_params = []
                    for name, tensor in self.model.named_parameters():
                        flag = False
                        for fel_name in freeze_enc_layers:
                            if fel_name in name:
                                frozen_params.append(tensor)
                                flag = True
                                break
                        if not flag:
                            optim_params.append(tensor)
                    print('num frozen_params: {}'.format(len(frozen_params)))
            self.optimizer = self.optm_fn(optim_params, lr=lr)

    def _create_dec_optimizer(self, weight_decay, lr):
        decay_parameters = get_parameter_names(self.model.tab_decoder, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "word_embed" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.tab_decoder.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.tab_decoder.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        self.dec_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

    def gradient_clip(self):
        if self.config.use_grad_clip:
            parameters = self.model.parameters()
            max_norm = self.config.max_grad_clip
            nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)

    def try_to_inject_classification_focal_loss(self, ori_model):
        if self.config.use_cls_focal:
            print('#--> Try to inject Focal Loss Calculator...')
            ori_model.register_classification_focal_loss_fn(
                self.config.focal_gamma
            )

    def train(self):
        log_interval = self.config.log_interval
        epoch_best_loss = 1e10
        epoch_best_valid_auc = -100
        epoch_best_valid_auc_epid = 0
        if self.valid_smaller_better:
            epoch_best_valid_score = 1e10
        else:
            epoch_best_valid_score = -1.0
        epoch_best_valid_score_epid = 0
        data_loader = None
        global_step = 0
        optimizer = self.optimizer
        n_accumulate_bp = 0
        early_stop_accum = 0
        for epid in range(self.num_epoch):
            self.model.train()
            print('=' * 40)
            print('####')
            print('#### Epoch {}'.format(epid))
            print('####')
            random.shuffle(self.train_files)
            total_step = 0
            total_loss = 0
            for train_file in self.train_files:
                if len(self.train_files) > 1 or data_loader is None:  # single dataset, no need reload
                    data_loader = self.build_data_loader(train_file, self.batch_size)
                for batch_wrapped_data in data_loader:
                    if global_step < self.freeze_encoder_steps:
                        optimizer = self.dec_optimizer
                    else:
                        if n_accumulate_bp > 0:
                            optimizer.step()
                            n_accumulate_bp = 0
                        optimizer = self.optimizer
                    global_step += 1
                    total_step += 1
                    if 0 == n_accumulate_bp:
                        optimizer.zero_grad()
                    loss = 0.0
                    loss_flag = False
                    for task in self.pretrain_task_names:
                        cur_task_data = batch_wrapped_data[task]
                        if cur_task_data is None:
                            continue
                        self.put_data_to_gpu(cur_task_data)
                        try:
                            result = self.model(cur_task_data, task)
                        except Exception as e:
                            if 'CUDA out of memory' in str(e):
                                print(e)
                                print('GPU OOM: SKIP THIS BATCH')
                                continue
                            else:
                                raise e
                        if torch.isnan(result['loss']):
                            print('** Loss is NaN, task: {}'.format(task))
                            continue
                        cur_loss = result['loss']
                        loss_flag = True
                        loss = loss + cur_loss * self.task_alphas[task]
                        log_vals = result['log_vals']
                        if self.task_train_logs[task] is None:
                            self.task_train_logs[task] = log_vals
                            log_vals['step'] = 1.0
                        else:
                            self.task_train_logs[task]['step'] += 1.0
                            for k, v in log_vals.items():
                                self.task_train_logs[task][k] += v
                    if loss_flag:
                        loss.backward()
                        self.gradient_clip()
                        total_loss += loss.item()
                        n_accumulate_bp += 1
                        if n_accumulate_bp >= self.n_g_accum:
                            n_accumulate_bp = 0
                            optimizer.step()
                    if 0 == total_step % log_interval:
                        print('-' * 60)
                        print('## epoch: {}, step: {}'.format(epid, total_step))
                        for task in self.pretrain_task_names:
                            cur_log_step = 1 if self.task_train_logs[task]['step'] == 0 else self.task_train_logs[task]['step']
                            cur_log_str = ','.join([' {}:{}'.format(k, round(v/cur_log_step, 6)) for k,v in self.task_train_logs[task].items() if k!='step'])
                            print('## {}|| {}'.format(task, cur_log_str))
                        self.reset_logs(self.task_train_logs)
                    if 0 == global_step % self.save_ckpt_interval:
                        print('## Saving newest ckpt ...')
                        if not self.no_save_raw:
                            torch.save(self.model, os.path.join(self.save_dir, 'Newest-EP-raw.pt'))
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'Newest-EP-state.pt'))
            if n_accumulate_bp > 0:
                n_accumulate_bp = 0
                optimizer.step()
            # save model here
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'Newest-EP-state.pt'))
            if not self.only_save_best:
                if not self.no_save_raw:
                    torch.save(self.model, os.path.join(self.save_dir, 'EP-raw-{}.pt'.format(epid)))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'EP-state-{}.pt'.format(epid)))
            epoch_avg_loss = total_loss / total_step
            if epoch_avg_loss < epoch_best_loss:
                epoch_best_loss = epoch_avg_loss
                print('## New Minimum Train Loss: {}'.format(epoch_best_loss))
                if not self.no_save_raw:
                    torch.save(self.model, os.path.join(self.save_dir, 'BEST-train-raw.pt'))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'BEST-train-state.pt'))
            if self.valid_loader is not None:
                valid_result = None
                if self.is_classification_tasks():
                    valid_result = self.valid()
                if self.is_common_predict_tasks():
                    valid_result = self.common_valid()
                if valid_result is not None and self.valid_metric in valid_result:
                    valid_score = valid_result[self.valid_metric]
                    is_new_valid_best = False
                    if self.valid_smaller_better:
                        if valid_score < epoch_best_valid_score:
                            epoch_best_valid_score = valid_score
                            epoch_best_valid_score_epid = epid
                            early_stop_accum = 0
                            is_new_valid_best = True
                        else:
                            early_stop_accum += 1
                    else:
                        if valid_score > epoch_best_valid_score:
                            epoch_best_valid_score = valid_score
                            epoch_best_valid_score_epid = epid
                            early_stop_accum = 0
                            is_new_valid_best = True
                        else:
                            early_stop_accum += 1
                    if is_new_valid_best:
                        print('FOUND NEW BEST VALID')
                        if not self.no_save_raw:
                            torch.save(self.model, os.path.join(self.save_dir, 'BEST-raw.pt'))
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'BEST-state.pt'))
                if self.early_stop and early_stop_accum > self.early_stop_patience:
                    print('Early Stop Now (tried {} times)'.format(self.early_stop_patience))
                    break
        print('Finished training.')
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'FINISHED-state.pt'))
        if self.valid_loader is not None:
            print('Best Valid SCORE: {}'.format(epoch_best_valid_score))
            print('Best Valid SCORE-Epoch Id: {}'.format(epoch_best_valid_score_epid))

    def train_each_task(self):
        """ each batch is only applied to individual task """
        log_interval = self.config.log_interval
        epoch_best_loss = 1e10
        epoch_best_valid_auc = -100
        epoch_best_valid_auc_epid = 0
        if self.valid_smaller_better:
            epoch_best_valid_score = 1e10
        else:
            epoch_best_valid_score = -1.0
        epoch_best_valid_score_epid = 0
        epoch_best_valid_loss = 1e10
        epoch_best_valid_loss_epid = 0
        data_loader = None
        global_step = 0
        optimizer = self.optimizer
        n_accumulate_bp = 0
        train_task_id = 0
        early_stop_accum = 0
        for epid in range(self.num_epoch):
            self.model.train()
            print('=' * 40)
            print('####')
            print('#### Epoch {}'.format(epid))
            print('####')
            random.shuffle(self.train_files)
            total_step = 0
            total_loss = 0
            for train_file in self.train_files:
                if len(self.train_files) > 1 or data_loader is None:  # single dataset, no need reload
                    data_loader = self.build_data_loader(train_file, self.batch_size)
                for batch_wrapped_data in data_loader:
                    if global_step < self.freeze_encoder_steps:
                        optimizer = self.dec_optimizer
                    else:
                        if n_accumulate_bp > 0:
                            optimizer.step()
                            n_accumulate_bp = 0
                        optimizer = self.optimizer
                    global_step += 1
                    total_step += 1
                    if 0 == n_accumulate_bp:
                        optimizer.zero_grad()
                    if train_task_id >= len(self.pretrain_task_names):
                        train_task_id = 0

                    task = self.pretrain_task_names[train_task_id]
                    cur_task_data = batch_wrapped_data[task]
                    if cur_task_data is None:
                        continue
                    self.put_data_to_gpu(cur_task_data)
                    try:
                        result = self.model(cur_task_data, task)
                    except Exception as e:
                        if 'CUDA out of memory' in str(e):
                            print(e)
                            print('GPU OOM: SKIP THIS BATCH')
                            continue
                        else:
                            raise e
                    if torch.isnan(result['loss']):
                        print('** Loss is NaN, task: {}'.format(task))
                        continue
                    cur_loss = result['loss']
                    loss = cur_loss * self.task_alphas[task]
                    log_vals = result['log_vals']
                    if self.task_train_logs[task] is None:
                        self.task_train_logs[task] = log_vals
                        log_vals['step'] = 1.0
                    else:
                        self.task_train_logs[task]['step'] += 1.0
                        for k, v in log_vals.items():
                            self.task_train_logs[task][k] += v

                    loss.backward()
                    self.gradient_clip()
                    total_loss += loss.item()
                    n_accumulate_bp += 1
                    train_task_id += 1
                    if n_accumulate_bp >= self.n_g_accum:
                        n_accumulate_bp = 0
                        optimizer.step()
                    if 0 == total_step % log_interval:
                        print('-' * 60)
                        print('## epoch: {}, step: {}'.format(epid, total_step))
                        for task in self.pretrain_task_names:
                            cur_log_step = 1 if self.task_train_logs[task]['step'] == 0 else self.task_train_logs[task]['step']
                            cur_log_str = ','.join([' {}:{}'.format(k, round(v/cur_log_step, 6)) for k,v in self.task_train_logs[task].items() if k!='step'])
                            print('## {}|| {}'.format(task, cur_log_str))
                        self.reset_logs(self.task_train_logs)
                    if 0 == global_step % self.save_ckpt_interval:
                        print('## Saving newest ckpt ...')
                        if not self.no_save_raw:
                            torch.save(self.model, os.path.join(self.save_dir, 'Newest-EP-raw.pt'))
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'Newest-EP-state.pt'))
            if n_accumulate_bp > 0:
                n_accumulate_bp = 0
                optimizer.step()
            # save model here
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'Newest-EP-state.pt'))
            if not self.only_save_best:
                if not self.no_save_raw:
                    torch.save(self.model, os.path.join(self.save_dir, 'EP-raw-{}.pt'.format(epid)))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'EP-state-{}.pt'.format(epid)))
            epoch_avg_loss = total_loss / total_step
            if epoch_avg_loss < epoch_best_loss:
                epoch_best_loss = epoch_avg_loss
                print('## New Minimum Train Loss: {}'.format(epoch_best_loss))
                if not self.no_save_raw:
                    torch.save(self.model, os.path.join(self.save_dir, 'BEST-train-raw.pt'))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'BEST-train-state.pt'))
            if self.valid_loader is not None:
                valid_result = None
                if self.is_classification_tasks():
                    valid_result = self.valid()
                if self.is_common_predict_tasks():
                    valid_result = self.common_valid()
                if valid_result is not None and self.valid_metric in valid_result:
                    valid_score = valid_result[self.valid_metric]
                    is_new_valid_best = False
                    if self.valid_smaller_better:
                        if valid_score < epoch_best_valid_score:
                            epoch_best_valid_score = valid_score
                            epoch_best_valid_score_epid = epid
                            early_stop_accum = 0
                            is_new_valid_best = True
                        else:
                            early_stop_accum += 1
                    else:
                        if valid_score > epoch_best_valid_score:
                            epoch_best_valid_score = valid_score
                            epoch_best_valid_score_epid = epid
                            early_stop_accum = 0
                            is_new_valid_best = True
                        else:
                            early_stop_accum += 1
                    if is_new_valid_best:
                        print('FOUND NEW BEST VALID')
                        if not self.no_save_raw:
                            torch.save(self.model, os.path.join(self.save_dir, 'BEST-raw.pt'))
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'BEST-state.pt'))
                if self.early_stop and early_stop_accum > self.early_stop_patience:
                    print('Early Stop Now (tried {} times)'.format(self.early_stop_patience))
                    break
        print('Finished training.')
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'FINISHED-state.pt'))
        if self.valid_loader is not None:
            print('Best Valid SCORE: {}'.format(epoch_best_valid_score))
            print('Best Valid SCORE-Epoch Id: {}'.format(epoch_best_valid_score_epid))

    def valid(self):
        print('validate ...')
        self.model.eval()
        old_dec_temperature = self.model.dec_temperature
        self.model.dec_temperature = 1.0
        metrics_record = {}
        avg_auc = []
        avg_acc = []
        for batch_wrapped_data in self.valid_loader:
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.model(cur_task_data, task)
                log_vals = result['log_vals']
                if self.task_valid_logs[task] is None:
                    self.task_valid_logs[task] = log_vals
                    log_vals['step'] = 1.0
                else:
                    self.task_valid_logs[task]['step'] += 1.0
                    for k, v in log_vals.items():
                        self.task_valid_logs[task][k] += v
                if 'classification_probs' in result and result['classification_probs'] is not None:
                    if task not in metrics_record:
                        metrics_record[task] = {
                            'y_prob_tgt': [],
                            'y_pred': [],
                            'y_label': []
                        }
                    if 2 == self.num_classification_labels:
                        pred_prob_of_tgt = result['classification_probs'][:, 1].tolist()
                    else:
                        pred_prob_of_tgt = result['classification_probs'].tolist()
                    metrics_record[task]['y_prob_tgt'] += pred_prob_of_tgt
                    metrics_record[task]['y_pred'] += result['classification_probs'].tolist()
                    metrics_record[task]['y_label'] += cur_task_data['batch_target_label'].tolist()
        for task in self.pretrain_task_names:
            cur_log_step = 1 if self.task_valid_logs[task]['step'] == 0 else self.task_valid_logs[task]['step']
            cur_log_str = ','.join(
                [' {}:{}'.format(k, round(v / cur_log_step, 6)) for k, v in self.task_valid_logs[task].items() if k != 'step']
            )
            print('## {}|| {}'.format(task, cur_log_str))
            if task in metrics_record:
                task_auc = evaluate(metrics_record[task]['y_prob_tgt'], metrics_record[task]['y_label'], metric='auc', seed=123)
                task_acc = evaluate(metrics_record[task]['y_pred'], metrics_record[task]['y_label'], metric='acc', seed=123)
                avg_auc += task_auc
                avg_acc += task_acc
        first_task = self.pretrain_task_names[0]
        cur_log_step = 1 if self.task_valid_logs[first_task]['step'] == 0 else self.task_valid_logs[first_task]['step']
        valid_result = {
            k: v / cur_log_step for k, v in self.task_valid_logs[first_task].items() if k != 'step'
        }
        valid_result['valid_auc'] = sum(avg_auc) / len(avg_auc) if len(avg_auc) > 0 else 0.0
        valid_result['valid_acc'] = sum(avg_acc) / len(avg_acc) if len(avg_acc) > 0 else 0.0
        self.reset_logs(self.task_valid_logs)
        self.model.train()
        self.model.dec_temperature = old_dec_temperature
        return valid_result

    def common_valid(self):
        print('common validate ...')
        self.model.eval()
        total_loss = 0.0
        total_step = 0.0
        for batch_wrapped_data in self.valid_loader:
            total_step += 1
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.model(cur_task_data, task)
                cur_loss = result['loss']
                total_loss += cur_loss.item() * self.task_alphas[task]
                log_vals = result['log_vals']
                if self.task_valid_logs[task] is None:
                    self.task_valid_logs[task] = log_vals
                    log_vals['step'] = 1.0
                else:
                    self.task_valid_logs[task]['step'] += 1.0
                    for k, v in log_vals.items():
                        self.task_valid_logs[task][k] += v
        for task in self.pretrain_task_names:
            cur_log_step = 1 if self.task_valid_logs[task]['step'] == 0 else self.task_valid_logs[task]['step']
            cur_log_str = ','.join(
                [' {}:{}'.format(k, round(v / cur_log_step, 6)) for k, v in self.task_valid_logs[task].items() if k != 'step']
            )
            print('## {}|| {}'.format(task, cur_log_str))
        first_task = self.pretrain_task_names[0]
        cur_log_step = 1 if self.task_valid_logs[first_task]['step'] == 0 else self.task_valid_logs[first_task]['step']
        valid_result = {
            k: v/cur_log_step for k, v in self.task_valid_logs[first_task].items() if k != 'step'
        }
        self.reset_logs(self.task_valid_logs)
        self.model.train()
        return valid_result

