# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

"""
multi gpu training version
"""

from unitab.model_builder import build_pretrain_model
from unitab.model_builder import build_data_collator
from unitab.tab_data_process import TokenizerProxy
from unitab.tab_data_loader import TabDataset
from torch.utils.data import DataLoader
import os
import random
import torch
from torch import nn
from unitab.train_utils import get_parameter_names, random_seed
from unitab.eval_utils import evaluate
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler


def build_model_and_tokenizer(config):
    random_seed(config.random_seed)
    tokenizer_proxy = TokenizerProxy(max_tok_len=config.max_tok_len)
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    tokenizer_proxy.save_tokenizer_to_disk(save_dir)

    pad_id = tokenizer_proxy.pad_token_id
    vocab_size = tokenizer_proxy.vocab_size
    model = build_pretrain_model(config, pad_id, vocab_size)
    total_params = sum([param.nelement() for param in model.parameters()])
    print('## total_params: {}'.format(total_params))
    total_enc_params = sum([param.nelement() for param in model.tab_encoder.parameters()])
    print('## total encoder params: {}'.format(total_enc_params))

    return model, tokenizer_proxy


class TabTrainer:
    def __init__(self, config, cur_local_rank):
        self.config = config
        self.cur_local_rank = cur_local_rank

        model, tokenizer_proxy = build_model_and_tokenizer(config)
        self.tokenizer_proxy = tokenizer_proxy
        self.missing_value_token = self.tokenizer_proxy.mask_token
        self.save_dir = config.save_dir

        if config.restore_path is not None:
            print('Restoring Parameters From: {}'.format(config.restore_path))
            try:
                state_dict = torch.load(config.restore_path, map_location='cpu')
                model.load_state_dict(state_dict)
            except Exception as e:
                # print(e)
                print('-' * 40)
                print('enter DistributedDataParallel model format restore...')
                state_dict = torch.load(config.restore_path, map_location='cpu')
                new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(os.path.join(self.save_dir, 'init-model-state.pt'), map_location='cpu')
            model.load_state_dict(state_dict)
        self.model = model.cuda()
        self.model = DDP(self.model, device_ids=[cur_local_rank], output_device=cur_local_rank, find_unused_parameters=True)

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
        if train_data_dir.endswith('.jsonl') and os.path.isfile(train_data_dir):
            train_files = [train_data_dir]
        else:
            files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
            train_files = [f for f in files if os.path.isfile(f) and f.endswith('.jsonl')]
        self.train_files = train_files
        self.num_epoch = config.num_epoch
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.task_train_logs = {tn: None for tn in self.pretrain_task_names}
        self.task_valid_logs = {tn: None for tn in self.pretrain_task_names}
        self.optimizer = None
        multi_gpu_ids = config.multi_gpu_ids
        num_gpus = len(multi_gpu_ids.split(','))
        self.lr = config.lr * num_gpus
        self._create_optimizer(config.weight_decay, config.lr)
        # self._create_dec_optimizer(config.weight_decay, config.lr)
        self.n_g_accum = config.n_g_accum
        self.valid_loader = None
        self.freeze_encoder_steps = config.freeze_encoder_steps
        assert self.freeze_encoder_steps < 1, 'pre-training do not support freezing parameters!!!!!!'
        self.save_ckpt_interval = config.save_ckpt_interval
        self.grad_scaler = GradScaler()
        self.use_mix_precision = config.use_f16
        self.no_save_raw = config.no_save_raw

    def build_data_loader(self, data_path, epoch_id, shuffle=True, drop_last=True, single_worker=False):
        # only used for train, no support for valid
        dataset = TabDataset(
            data_path, self.missing_value_token,
            reader_num_processes=self.config.reader_num_processes,
            num_precision=self.config.num_precision,
            max_num_examples=self.config.trunc_loading_max_num_examples
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_sampler.set_epoch(epoch_id)
        shuffle = False  # INFO sampler is mutual to shuffle, and will occur errors
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_data_collate_fn,
            drop_last=drop_last,
            num_workers=self.num_workers if not single_worker else 1,
            sampler=train_sampler
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
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

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

    def is_main_node(self):
        return 0 == self.cur_local_rank

    def save_model(self, ckpt_name, is_state):
        if is_state:
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, ckpt_name))
        else:
            torch.save(self.model.module, os.path.join(self.save_dir, ckpt_name))

    def call_model(self, cur_task_data, task):
        if self.use_mix_precision:
            with autocast():
                result = self.model(cur_task_data, task)
        else:
            result = self.model(cur_task_data, task)
        return result

    def backward_loss(self, loss):
        if self.use_mix_precision:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def bp_step(self, optimizer):
        if self.use_mix_precision:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()

    def train(self):
        is_main_node = self.is_main_node()
        log_interval = self.config.log_interval
        epoch_best_loss = 1e10
        epoch_best_auc = -100
        epoch_best_auc_epid = 0
        data_loader = None
        global_step = 0
        optimizer = self.optimizer
        n_accumulate_bp = 0
        for epid in range(self.num_epoch):
            self.model.train()
            print('=' * 40)
            print('####')
            print('#### Epoch {}'.format(epid))
            print('####')
            random.shuffle(self.train_files)
            total_step = 0
            total_loss = 0
            for train_file_id, train_file in enumerate(self.train_files):
                print('train_file_id: {} / {}'.format(train_file_id, len(self.train_files)))
                if len(self.train_files) > 1 or data_loader is None:  # single dataset, no need reload
                    if data_loader is not None:
                        del data_loader
                    data_loader = self.build_data_loader(train_file, epid)
                for batch_wrapped_data in data_loader:
                    if global_step < self.freeze_encoder_steps:
                        optimizer = self.dec_optimizer
                    else:
                        if n_accumulate_bp > 0:
                            # optimizer.step()
                            self.bp_step(optimizer)
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
                            result = self.call_model(cur_task_data, task)
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
                        # loss.backward()
                        self.backward_loss(loss)
                        self.gradient_clip()
                        total_loss += loss.item()
                        n_accumulate_bp += 1
                        if n_accumulate_bp >= self.n_g_accum:
                            n_accumulate_bp = 0
                            # optimizer.step()
                            self.bp_step(optimizer)
                    if 0 == total_step % log_interval:
                        print('-' * 60)
                        print('## epoch: {}, step: {}'.format(epid, total_step))
                        for task in self.pretrain_task_names:
                            cur_log_step = 1 if self.task_train_logs[task]['step'] == 0 else self.task_train_logs[task]['step']
                            cur_log_str = ','.join([' {}:{}'.format(k, round(v/cur_log_step, 6)) for k,v in self.task_train_logs[task].items() if k!='step'])
                            print('## {}|| {}'.format(task, cur_log_str))
                        self.reset_logs(self.task_train_logs)
                    if is_main_node and 0 == global_step % self.save_ckpt_interval:
                        print('## Saving newest ckpt ...')
                        if not self.no_save_raw:
                            self.save_model('Newest-EP-raw.pt', False)
                        self.save_model('Newest-EP-state.pt', True)
                        if not self.no_save_raw:
                            self.save_model('Inner-EP-raw-{}.pt'.format(global_step), False)
                        self.save_model('Inner-EP-state-{}.pt'.format(global_step), True)
            if n_accumulate_bp > 0:
                n_accumulate_bp = 0
                # optimizer.step()
                self.bp_step(optimizer)
            # save model here
            if is_main_node:
                if not self.no_save_raw:
                    self.save_model('EP-raw-{}.pt'.format(epid), False)
                self.save_model('EP-state-{}.pt'.format(epid), True)
                epoch_avg_loss = total_loss / total_step
                if epoch_avg_loss < epoch_best_loss:
                    epoch_best_loss = epoch_avg_loss
                    print('## New Minimum Train Loss: {}'.format(epoch_best_loss))
                    if not self.no_save_raw:
                        self.save_model('BEST-raw.pt', False)
                    self.save_model('BEST-state.pt', True)
            if self.valid_loader is not None:
                valid_auc = self.valid()
                if valid_auc is not None and valid_auc > epoch_best_auc:
                    epoch_best_auc = valid_auc
                    epoch_best_auc_epid = epid
        print('Finished training.')
        if self.valid_loader is not None:
            print('Best Valid AUC: {}'.format(epoch_best_auc))
            print('Best Valid AUC-Epoch Id: {}'.format(epoch_best_auc_epid))

    def train_each_task(self):
        """ each batch is only applied to individual task """
        is_main_node = self.is_main_node()
        log_interval = self.config.log_interval
        epoch_best_loss = 1e10
        epoch_best_auc = -100
        epoch_best_auc_epid = 0
        data_loader = None
        global_step = 0
        optimizer = self.optimizer
        n_accumulate_bp = 0
        train_task_id = 0
        for epid in range(self.num_epoch):
            self.model.train()
            print('=' * 40)
            print('####')
            print('#### Epoch {}'.format(epid))
            print('####')
            random.shuffle(self.train_files)
            total_step = 0
            total_loss = 0
            for train_file_id, train_file in enumerate(self.train_files):
                print('train_file_id: {} / {}'.format(train_file_id, len(self.train_files)))
                if len(self.train_files) > 1 or data_loader is None:  # single dataset, no need reload
                    if data_loader is not None:
                        del data_loader
                    data_loader = self.build_data_loader(train_file, epid)
                for batch_wrapped_data in data_loader:
                    if global_step < self.freeze_encoder_steps:
                        optimizer = self.dec_optimizer
                    else:
                        if n_accumulate_bp > 0:
                            # optimizer.step()
                            self.bp_step(optimizer)
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
                        result = self.call_model(cur_task_data, task)
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

                    # loss.backward()
                    self.backward_loss(loss)
                    self.gradient_clip()
                    total_loss += loss.item()
                    n_accumulate_bp += 1
                    train_task_id += 1
                    if n_accumulate_bp >= self.n_g_accum:
                        n_accumulate_bp = 0
                        # optimizer.step()
                        self.bp_step(optimizer)
                    if 0 == total_step % log_interval:
                        print('-' * 60)
                        print('## epoch: {}, step: {}'.format(epid, total_step))
                        for task in self.pretrain_task_names:
                            if self.task_train_logs[task] is not None:
                                cur_log_step = 1 if self.task_train_logs[task]['step'] == 0 else self.task_train_logs[task]['step']
                                cur_log_str = ','.join([' {}:{}'.format(k, round(v/cur_log_step, 6)) for k,v in self.task_train_logs[task].items() if k!='step'])
                                print('## {}|| {}'.format(task, cur_log_str))
                        self.reset_logs(self.task_train_logs)
                    if is_main_node and 0 == global_step % self.save_ckpt_interval:
                        print('## Saving newest ckpt ...')
                        if not self.no_save_raw:
                            self.save_model('Newest-EP-raw.pt', False)
                        self.save_model('Newest-EP-state.pt', True)
                        if not self.no_save_raw:
                            self.save_model('Inner-EP-raw-{}.pt'.format(global_step), False)
                        self.save_model('Inner-EP-state-{}.pt'.format(global_step), True)
            if n_accumulate_bp > 0:
                n_accumulate_bp = 0
                # optimizer.step()
                self.bp_step(optimizer)
            # save model here
            if is_main_node:
                if not self.no_save_raw:
                    self.save_model('EP-raw-{}.pt'.format(epid), False)
                self.save_model('EP-state-{}.pt'.format(epid), True)
                epoch_avg_loss = total_loss / total_step
                if epoch_avg_loss < epoch_best_loss:
                    epoch_best_loss = epoch_avg_loss
                    print('## New Minimum Train Loss: {}'.format(epoch_best_loss))
                    if not self.no_save_raw:
                        self.save_model('BEST-raw.pt', False)
                    self.save_model('BEST-state.pt', True)
            if self.valid_loader is not None:
                valid_auc = self.valid()
                if valid_auc is not None and valid_auc > epoch_best_auc:
                    epoch_best_auc = valid_auc
                    epoch_best_auc_epid = epid
        print('Finished training.')
        if self.valid_loader is not None:
            print('Best Valid AUC: {}'.format(epoch_best_auc))
            print('Best Valid AUC-Epoch Id: {}'.format(epoch_best_auc_epid))

    def train_bp_immediately(self):
        is_main_node = self.is_main_node()
        log_interval = self.config.log_interval
        epoch_best_loss = 1e10
        epoch_best_auc = -100
        epoch_best_auc_epid = 0
        data_loader = None
        global_step = 0
        optimizer = self.optimizer
        n_accumulate_bp = 0
        for epid in range(self.num_epoch):
            self.model.train()
            print('=' * 40)
            print('####')
            print('#### Epoch {}'.format(epid))
            print('####')
            random.shuffle(self.train_files)
            total_step = 0
            total_loss = 0
            for train_file_id, train_file in enumerate(self.train_files):
                print('train_file_id: {} / {}'.format(train_file_id, len(self.train_files)))
                if len(self.train_files) > 1 or data_loader is None:  # single dataset, no need reload
                    if data_loader is not None:
                        del data_loader
                    data_loader = self.build_data_loader(train_file, epid)
                for batch_wrapped_data in data_loader:
                    if global_step < self.freeze_encoder_steps:
                        optimizer = self.dec_optimizer
                    else:
                        if n_accumulate_bp > 0:
                            # optimizer.step()
                            self.bp_step(optimizer)
                            n_accumulate_bp = 0
                        optimizer = self.optimizer
                    global_step += 1
                    total_step += 1
                    if 0 == n_accumulate_bp:
                        optimizer.zero_grad()
                    cur_step_total_loss_val = 0.0
                    loss_flag = False
                    for task in self.pretrain_task_names:
                        cur_task_data = batch_wrapped_data[task]
                        if cur_task_data is None:
                            continue
                        self.put_data_to_gpu(cur_task_data)
                        try:
                            result = self.call_model(cur_task_data, task)
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
                        cur_bp_loss = cur_loss * self.task_alphas[task]
                        self.backward_loss(cur_bp_loss)
                        self.gradient_clip()
                        loss_flag = True
                        cur_step_total_loss_val += cur_bp_loss.item()
                        log_vals = result['log_vals']
                        if self.task_train_logs[task] is None:
                            self.task_train_logs[task] = log_vals
                            log_vals['step'] = 1.0
                        else:
                            self.task_train_logs[task]['step'] += 1.0
                            for k, v in log_vals.items():
                                self.task_train_logs[task][k] += v
                    if loss_flag:
                        total_loss += cur_step_total_loss_val
                        n_accumulate_bp += 1
                        if n_accumulate_bp >= self.n_g_accum:
                            n_accumulate_bp = 0
                            self.bp_step(optimizer)
                    if 0 == total_step % log_interval:
                        print('-' * 60)
                        print('## epoch: {}, step: {}'.format(epid, total_step))
                        for task in self.pretrain_task_names:
                            cur_log_step = 1 if self.task_train_logs[task]['step'] == 0 else self.task_train_logs[task]['step']
                            cur_log_str = ','.join([' {}:{}'.format(k, round(v/cur_log_step, 6)) for k,v in self.task_train_logs[task].items() if k!='step'])
                            print('## {}|| {}'.format(task, cur_log_str))
                        self.reset_logs(self.task_train_logs)
                    if is_main_node and 0 == global_step % self.save_ckpt_interval:
                        print('## Saving newest ckpt ...')
                        if not self.no_save_raw:
                            self.save_model('Newest-EP-raw.pt', False)
                        self.save_model('Newest-EP-state.pt', True)
                        if not self.no_save_raw:
                            self.save_model('Inner-EP-raw-{}.pt'.format(global_step), False)
                        self.save_model('Inner-EP-state-{}.pt'.format(global_step), True)
            if n_accumulate_bp > 0:
                n_accumulate_bp = 0
                # optimizer.step()
                self.bp_step(optimizer)
            # save model here
            if is_main_node:
                print('## Saving epoch ckpt ...')
                if not self.no_save_raw:
                    self.save_model('EP-raw-{}.pt'.format(epid), False)
                self.save_model('EP-state-{}.pt'.format(epid), True)
                epoch_avg_loss = total_loss / total_step
                if epoch_avg_loss < epoch_best_loss:
                    epoch_best_loss = epoch_avg_loss
                    print('## New Minimum Train Loss: {}'.format(epoch_best_loss))
                    if not self.no_save_raw:
                        self.save_model('BEST-raw.pt', False)
                    self.save_model('BEST-state.pt', True)
            if self.valid_loader is not None:
                valid_auc = self.valid()
                if valid_auc is not None and valid_auc > epoch_best_auc:
                    epoch_best_auc = valid_auc
                    epoch_best_auc_epid = epid
        print('Finished training.')
        if self.valid_loader is not None:
            print('Best Valid AUC: {}'.format(epoch_best_auc))
            print('Best Valid AUC-Epoch Id: {}'.format(epoch_best_auc_epid))

    def valid(self):
        print('validate ...')
        self.model.eval()
        metrics_record = {}
        avg_auc = []
        for batch_wrapped_data in self.valid_loader:
            for task in self.pretrain_task_names:
                cur_task_data = batch_wrapped_data[task]
                if cur_task_data is None:
                    continue
                self.put_data_to_gpu(cur_task_data)
                result = self.call_model(cur_task_data, task)
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
                    # pred_prob_of_tgt = result['classification_probs'].gather(1, cur_task_data['batch_target_label'].unsqueeze(1)).squeeze(1).tolist()
                    # 临时针对 2分类, 多分类在计算调用 roc_auc_score(Y_test, Y_pred_prob, multi_class='ovo'), Y_pred_prob shape [N, num_classes]
                    pred_prob_of_tgt = result['classification_probs'][:, 1].tolist()
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
                evaluate(metrics_record[task]['y_pred'], metrics_record[task]['y_label'], metric='acc', seed=123)
                avg_auc += task_auc
        self.reset_logs(self.task_valid_logs)
        self.model.train()
        if len(avg_auc) < 1:
            return None
        return sum(avg_auc) / len(avg_auc)


def set_distribution_train_config():
    import argparse
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    working_dir = os.environ['WORK_DIR']  # INFO we need to put working dir to os environment params
    print('## working dir: {}'.format(working_dir))
    config_path = os.path.join(working_dir, 'train_config.pt')
    train_config = torch.load(config_path)

    device_config = set_distribution_train_config()
    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    world_size = torch.cuda.device_count()
    local_rank = device_config.local_rank
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    if 0 == local_rank:
        import streamtologger
        streamtologger.redirect(
            target=os.path.join(working_dir, 'log_train.txt'), append=True,
            header_format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] "
        )

    trainer = TabTrainer(train_config, local_rank)
    train_mode = train_config.train_mode
    if 'default' == train_mode:
        trainer.train()
    elif 'sep_each_task' == train_mode:
        trainer.train_each_task()
    elif 'default_bp_each_task' == train_mode:
        trainer.train_bp_immediately()
    else:
        trainer.train()

    if 0 == local_rank:
        print('Finished, exiting ...')
        dist.destroy_process_group()



