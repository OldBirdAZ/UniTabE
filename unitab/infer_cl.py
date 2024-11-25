# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import math
import torch
import json
from unitab.tab_data_process import TokenizerProxy
from unitab.model_builder import build_pretrain_model
from unitab.model_builder import build_data_collator
from unitab.tab_data_loader import wrap_each_sample

COLUMN_SEP = ' | '


def preprocess_example(example_str, data_meta):
    kvs = example_str.split(COLUMN_SEP)
    raw_kvs = {}
    for kvpair in kvs:
        ckp = kvpair.split(':')
        raw_kvs[ckp[0]] = ckp[1]
    kv_s = {}
    num_features = []
    bin_features = []
    for cn, ct in data_meta.items():
        if 'string' == ct:
            default_cv = None
        elif 'number' == ct:
            default_cv = math.nan
        elif 'binary' == ct:
            default_cv = None
        else:
            raise ValueError('Illegal column type')
        kv_s[cn] = default_cv
        if cn in raw_kvs:
            if 'string' == ct:
                cv = str(raw_kvs[cn])
            elif 'number' == ct:
                cv = float(raw_kvs[cn])
            elif 'binary' == ct:
                cv = str(raw_kvs[cn])
            else:
                raise ValueError('Illegal column type')
            kv_s[cn] = cv
    item = {
        'data': {
            'kv_s': kv_s,
            'target_label': 0,
            'num_features': num_features,
            'bin_features': bin_features,
            'example_id': 0,
        },
        'description': 'dataset name'
    }
    return item


class TabInferCL:
    def __init__(self, infer_config):
        #print('--> loading model & setting ...')
        data_meta_path = infer_config.data_meta_path
        self.data_meta = json.load(open(data_meta_path))
        self.train_config = torch.load(infer_config.train_config_path)
        self.tokenizer_proxy = TokenizerProxy(max_tok_len=self.train_config.max_tok_len, verbose=False)
        self.task_name = infer_config.task_name
        pretrain_task_names = 'pure_cls3_task'
        if 'classification' == infer_config.task_name:
            pretrain_task_names = 'pure_cls3_task'
        elif 'common_predict' == infer_config.task_name:
            pretrain_task_names = 'common_predict_sep_answer_task'
        else:
            raise ValueError('Illegal task_name')
        self.original_task_names = [pretrain_task_names]
        self.batch_data_collate_fn = build_data_collator(self.train_config, self.tokenizer_proxy, self.original_task_names, verbose=False)
        pad_id = self.tokenizer_proxy.pad_token_id
        vocab_size = self.tokenizer_proxy.vocab_size
        self.missing_value_token = self.tokenizer_proxy.mask_token
        model = build_pretrain_model(self.train_config, pad_id, vocab_size)
        state_dict = torch.load(infer_config.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        self.model = model
        self.model.eval()
        self.use_gpu = not infer_config.no_gpu
        if self.use_gpu:
            self.model.cuda()
        self.data_meta = json.load(open(infer_config.data_meta_path))
        self.test_max_dec_steps = infer_config.test_max_dec_steps

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

    def do_cl_infer(self, example_str):
        # print('--> processing data ...')
        item = preprocess_example(example_str, self.data_meta)
        pitem = wrap_each_sample(item, self.missing_value_token)
        pitem['raw_data'] = example_str
        cur_task_data = self.batch_data_collate_fn([pitem])[self.original_task_names[0]]
        if self.use_gpu:
            self.put_data_to_gpu(cur_task_data)
        # print('--> predicting ...')
        if 'classification' == self.task_name:
            result = self.model(cur_task_data, self.original_task_names[0])
            pred_result = torch.argmax(result['classification_probs'], 1).tolist()[0]
        elif 'common_predict' == self.task_name:
            result = self.model.infer_common(cur_task_data, max_dec_steps=self.test_max_dec_steps)
            dec_out_ids = result['dec_out_ids']
            batch_dec_texts = dec_ids_to_str(dec_out_ids, self.tokenizer_proxy)
            pred_result = batch_dec_texts[0]
        #print('## pred_result:')
        return pred_result


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


