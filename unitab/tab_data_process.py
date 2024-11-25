# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import os
from transformers import BertTokenizerFast
import random
import math
import numpy as np
import torch
import copy
from unitab import common_const


class TokenizerProxy:
    def __init__(self, disable_tokenizer_parallel=False, max_tok_len=512, verbose=False):
        """
        :param disable_tokenizer_parallel:
        :param max_tok_len: if len(toks) > max_tok_len, then trunc it to be max_tok_len
        """
        if os.path.exists('./unitab/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./unitab/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./unitab/tokenizer')
        # self.tokenizer.__dict__['model_max_length'] = 512
        self.tokenizer.__dict__['model_max_length'] = max_tok_len
        if disable_tokenizer_parallel:  # disable tokenizer parallel # 实验设置中该选项为False
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token
        self.pad_token = self.tokenizer.pad_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        # self.bos_token_id = self.tokenizer.bos_token_id  # bert中没用用到
        # self.eos_token_id = self.tokenizer.eos_token_id  # bert中没用用到
        self.bos_token = '[unused0]'
        self.eos_token = '[unused1]'
        self.missing_value_token = '[unused2]'
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.missing_value_token_id = self.tokenizer.convert_tokens_to_ids(self.missing_value_token)
        if verbose:
            print('#' * 20)
            print('mask: {}-{}'.format(self.mask_token, self.mask_token_id))
            print('pad: {}-{}'.format(self.pad_token, self.pad_token_id))
            print('sep: {}-{}'.format(self.sep_token, self.sep_token_id))
            print('cls: {}-{}'.format(self.cls_token, self.cls_token_id))
            print('bos: {}-{}'.format(self.bos_token, self.bos_token_id))
            print('eos: {}-{}'.format(self.eos_token, self.eos_token_id))
            print('missing: {}-{}'.format(self.missing_value_token, self.missing_value_token_id))
            print('#' * 20)

    def save_tokenizer_to_disk(self, save_dir):
        self.tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))

    def tokenize_to_tensor(self, str_or_list_of_strs, max_length=None):
        """
        :param str_or_list_of_strs: str or list of str
        :return: PyTorch Tensor
        """
        tok_results = self.tokenizer(
            str_or_list_of_strs, padding=True, truncation=True,
            add_special_tokens=False, return_tensors='pt',
            max_length=max_length
        )
        return tok_results

    def tokenize_to_numpy(self, str_or_list_of_strs, max_length=None):
        """
        :param str_or_list_of_strs: str or list of str
        :return: Dict
            'hello world!' --> {'input_ids': array([[7592, 2088,  999]]), 'token_type_ids': array([[0, 0, 0]]), 'attention_mask': array([[1, 1, 1]])}
        """
        tok_results = self.tokenizer(
            str_or_list_of_strs, padding=True, truncation=True,
            add_special_tokens=False, return_tensors='np',
            max_length=max_length
        )
        return tok_results

    def token2id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def decode_ids(self, ids):
        return self.tokenizer.decode(ids)


class TabularPretrainCollator:
    """support positive pair sampling for contrastive learning of transtab model.
        + recovery
    """
    def __init__(self, tokenizer_proxy, task_names, overlap_ratio=0.5, num_partition=3, num_classification_labels=2,
                 max_num_features=-1, max_feature_tok_len=None, max_prompt_tok_len=None,
                 common_pred_prefix='predict label :', test_mode=False, test_max_dec_steps=1, verbose=True,
                 data_config=None
                 ):
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition, int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.data_config = data_config
        assert self.data_config is not None
        self.overlap_ratio = overlap_ratio
        self.num_partition = num_partition
        self.tokenizer_proxy = tokenizer_proxy
        self.task_names = task_names
        self.max_num_features = max_num_features
        self.max_feature_tok_len = max_feature_tok_len
        self.max_prompt_tok_len = max_prompt_tok_len
        self._filter_num_features = self.max_num_features > 0
        self.common_pred_prefix = common_pred_prefix
        self.test_mode = test_mode
        self.test_max_dec_steps = test_max_dec_steps
        assert isinstance(self.tokenizer_proxy, TokenizerProxy)
        self.mask_token = self.tokenizer_proxy.mask_token
        self.eos_token = self.tokenizer_proxy.eos_token
        self.missing_value_token = self.tokenizer_proxy.missing_value_token
        self.eos_token_id = self.tokenizer_proxy.eos_token_id
        self.verbose = verbose

        # inject config for each task here
        self._inject_conf_for_common_classification()
        self._inject_conf_for_dynamic_mask_span()
        self._inject_conf_for_mask_fill_missval_predict()

        self.task_register = {
            common_const.TASK_NAME_CLASSIFICATION: self.build_for_classification3,
            common_const.TASK_NAME_COMMON_PREDICTION: self.build_for_common_predict,
            common_const.TASK_NAME_COMMON_PREDICTION_SEP_ANSWER: self.build_for_common_predict_sep_answer,
            common_const.TASK_NAME_CONTRASTIVE_LEARN: self.build_for_scl,
            common_const.TASK_NAME_CONTRASTIVE_LEARN2: self.build_for_scl,
            common_const.TASK_NAME_MASK_RECOVERY: self.build_for_mask_recovery,
            common_const.TASK_NAME_DYNAMIC_MASK_SPAN_RECOVERY: self.build_for_mask_recovery_dynamic_span,
            common_const.TASK_NAME_RECALL: self.build_for_recall,
            common_const.TASK_NAME_WHICH_IS_BIGGER: self.build_for_which_is_bigger,
            common_const.TASK_NAME_DATA_TYPE: self.build_for_predict_data_type,
            common_const.TASK_NAME_WHETHER_IS_MISSING_VALUE: self.build_for_whether_is_missing_value,
            common_const.TASK_NAME_FILL_MISSING_VALUE: self.build_for_fill_missing_value,
            common_const.TASK_NAME_COMMON_FILL_MISSVAL_PREDICTION: self.build_for_mask_fill_missval_predict,
        }
        self.share_basic_data_tasks = [
            common_const.TASK_NAME_CLASSIFICATION,
            common_const.TASK_NAME_COMMON_PREDICTION,
            common_const.TASK_NAME_COMMON_PREDICTION_SEP_ANSWER,
            common_const.TASK_NAME_RECALL,
            common_const.TASK_NAME_DATA_TYPE,
        ]
        self.has_share_task = len(set(task_names) & set(self.share_basic_data_tasks)) > 0

        self.num_classification_labels = num_classification_labels
        self.classification_option_template = '[unused{}]'
        self.classification_suffix = ' '.join(['[unused{}]'.format(int(10 + lidx)) for lidx in range(self.num_classification_labels)])

        classification_labels_to_tokids = {}
        classification_options = []
        classification_options_infone = []
        for lidx in range(self.num_classification_labels):
            c_k = self.classification_option_template.format(int(10 + lidx))
            c_v = self.tokenizer_proxy.tokenizer.convert_tokens_to_ids(c_k)
            classification_labels_to_tokids[lidx] = {'label_token': c_k, 'label_token_id': c_v}
            classification_options.append(c_k)
            classification_options_infone.append(c_v)
        self.classification_labels_to_tokids = classification_labels_to_tokids
        self.classification_options_infone = classification_options_infone
        classification_prompt_prefix = 'classification task , the options are {}'.format(' '.join(classification_options))
        classification_prompt2_prefix = 'classification task , the options are {} , then the answer is'.format(' '.join(classification_options))
        classification_prompt3_prefix = 'classification :'
        self.classification_prompt_prefix_ids = self.tokenizer_proxy.tokenizer.convert_tokens_to_ids(classification_prompt_prefix.split(' '))
        self.classification_prompt2_prefix_ids = self.tokenizer_proxy.tokenizer.convert_tokens_to_ids(classification_prompt2_prefix.split(' '))
        self.classification_prompt3_prefix_ids = self.tokenizer_proxy.tokenizer.convert_tokens_to_ids(classification_prompt3_prefix.split(' '))
        if verbose:
            print('classification_labels_to_tokids: {}'.format(classification_labels_to_tokids))
            print('classification_prompt_prefix_ids: {}'.format(self.classification_prompt_prefix_ids))
            print('classification_prompt2_prefix_ids: {}'.format(self.classification_prompt2_prefix_ids))
            print('classification_prompt3_prefix_ids: {}'.format(self.classification_prompt3_prefix_ids))

    def _inject_conf_for_common_classification(self):
        common_classification_prefix = self.data_config.common_classification_prefix
        self.common_classification_prefix_ids = self.tokenizer_proxy.tokenizer.convert_tokens_to_ids(
            common_classification_prefix.split(' ')
        )

    def _inject_conf_for_dynamic_mask_span(self):
        dynamic_mask_missval_span = self.data_config.dynamic_mask_missval_span
        pairs = [p for p in dynamic_mask_missval_span.split('|')]
        th_pairs = []
        for p in pairs:
            pp = p.split(':')
            th_pairs.append((float(pp[0]), int(pp[1])))
        self.dynamic_mask_missval_span = th_pairs
        self.dynamic_mask_span_max_th = self.data_config.dynamic_mask_span_max_th

    def _inject_conf_for_mask_fill_missval_predict(self):
        self._common_fill_missval_column = self.data_config.common_fill_missval_column
        self._gen_constrained_toks_ids = None
        gen_constrained_toks = self.data_config.gen_constrained_toks
        gen_constrained_sep = self.data_config.gen_constrained_sep
        if gen_constrained_toks is not None:
            gen_constrained_toks = gen_constrained_toks.split(gen_constrained_sep)
            self._gen_constrained_toks_ids = [self.tokenizer_proxy.token2id(tok) for tok in gen_constrained_toks]
        if self.verbose:
            print('_gen_constrained_toks_ids:: {}'.format(self._gen_constrained_toks_ids))

    def _build_basic_data(self, batch_data_list):
        """
        classification
        :param batch_data_list: List
        :return:
        """
        basic_data = wrap_subset_data([item['features'] for item in batch_data_list], self.tokenizer_proxy, self.max_feature_tok_len)
        return basic_data

    def build_for_scl(self, batch_data_list, shared_basic_data):
        """
        self supervised learning: contrastive learning
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        select_ids = [idx_ for idx_ in range(self.num_partition if self.num_partition>1 else 2)]
        random.shuffle(select_ids)
        pos_pair_idx1 = select_ids[0]
        pos_pair_idx2 = select_ids[1]
        p1_list = []
        p2_list = []
        for bidx, item in enumerate(batch_data_list):
            column_names = list(item['features'].keys())
            if self.num_partition > 1:
                sub_x_list = _build_positive_pairs(column_names, self.num_partition, self.overlap_ratio)
            else:
                sub_x_list = _build_positive_pairs_single_view(column_names)
            if len(sub_x_list) > 2:
                p1_list.append({
                    cn: item['features'][cn] for cn in sub_x_list[pos_pair_idx1]
                })
                p2_list.append({
                    cn: item['features'][cn] for cn in sub_x_list[pos_pair_idx2]
                })
            elif len(sub_x_list) == 2:
                p1_list.append({
                    cn: item['features'][cn] for cn in sub_x_list[0]
                })
                p2_list.append({
                    cn: item['features'][cn] for cn in sub_x_list[1]
                })
        if len(p1_list) < 2:
            return None
        p1_basic_data = wrap_subset_data(p1_list, self.tokenizer_proxy, self.max_feature_tok_len)
        p2_basic_data = wrap_subset_data(p2_list, self.tokenizer_proxy, self.max_feature_tok_len)
        p1_processed_data = {'basic_data': p1_basic_data}
        p2_processed_data = {'basic_data': p2_basic_data}

        batch_prompt = []
        batch_size = len(p1_list)
        neg_pair_ids = []
        for i in range(batch_size):
            pidx = random.randint(1, batch_size - 1)
            neg_pair_ids.append((i + pidx) % batch_size)
            batch_prompt.append('semantic representation :')
        neg_pair_ids = torch.LongTensor(neg_pair_ids)
        p1_processed_data['neg_pair_ids'] = neg_pair_ids
        prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt, self.max_prompt_tok_len)
        batch_prompt_ids = prompts_tok['input_ids']
        batch_prompt_mask = prompts_tok['attention_mask']
        p1_processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        p1_processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        datas = [p1_processed_data, p2_processed_data]
        return datas

    def build_for_classification3(self, batch_data_list, shared_basic_data):
        """
        classification
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        # avoid single classification options is convert to multiple token-ids
        lables = [item['target_label'] for item in batch_data_list]
        processed_data = {'basic_data': shared_basic_data}
        if self.test_mode:
            if -1 in lables:  # we need to reset to false label so that it will not throw exception
                lables = [0] * len(lables)
        processed_data['batch_target_label'] = torch.LongTensor(lables)
        processed_data['num_classification_labels'] = self.num_classification_labels
        batch_prompt_prefix_ids = []
        batch_prompt_prefix_mask = []
        prompt_prefix_mask = [1] * len(self.common_classification_prefix_ids)
        batch_raw_data = []
        batch_value_bos_ids = []
        batch_value_mask = []
        bos_token_id = self.tokenizer_proxy.bos_token_id
        for bidx, cur_label in enumerate(lables):
            batch_prompt_prefix_ids.append(self.common_classification_prefix_ids)
            batch_prompt_prefix_mask.append(prompt_prefix_mask)
            batch_raw_data.append(batch_data_list[bidx]['raw_data'])
            batch_value_bos_ids.append([bos_token_id])
            batch_value_mask.append([1])
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        processed_data['batch_value_bos_ids'] = torch.LongTensor(batch_value_bos_ids)
        processed_data['batch_value_eos_ids'] = torch.LongTensor(batch_value_bos_ids)
        processed_data['batch_value_mask'] = torch.BoolTensor(batch_value_mask)
        if self.test_mode:
            processed_data['batch_raw_data'] = batch_raw_data
            processed_data['batch_raw_target'] = lables
        infone_options = [self.classification_options_infone]
        # [1, num_classification_labels]
        processed_data['infone_ids'] = torch.LongTensor(infone_options)
        processed_data['infone_mask'] = torch.BoolTensor([[1] * len(self.classification_options_infone)])
        return processed_data

    def build_for_mask_recovery(self, batch_data_list, shared_basic_data):
        """
        mask value of single cell for each example --> recovery its value
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        ori_cols = []
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        batch_prompt_value = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            column_names = list(cur_row.keys())
            no_missing_col_names = [k for k,v in cur_row.items() if not v['is_missing']]
            if len(no_missing_col_names) < 1:
                continue
            pidx = random.randint(0, len(no_missing_col_names) - 1)
            selected_col = no_missing_col_names[pidx]
            selected_col_value = cur_row[selected_col]
            ori_cols.append((selected_col, selected_col_value))
            new_features = {cn: cur_row[cn] for cn in column_names}
            new_features[selected_col] = copy.deepcopy(selected_col_value)
            new_features[selected_col]['processed_value'] = '{}'.format(self.mask_token)
            batch_row_features.append(new_features)

            prompt = 'recovery {} : {} {}'.format(selected_col, selected_col_value['processed_value'], self.eos_token)
            prompt_prefix = 'recovery {} :'.format(selected_col)
            prompt_value = '{}'.format(selected_col_value['processed_value'])
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
            batch_prompt_value.append(prompt_value)
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_pprefix_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_prefix, self.max_prompt_tok_len)
        tmp_pvalue_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_value, self.max_feature_tok_len)
        batch_prompt_ids, batch_prompt_mask, batch_prompt_prefix_ids, batch_prompt_prefix_mask = merge_prompt_and_value(
            tmp_pprefix_tok, tmp_pvalue_tok, self.eos_token_id
        )
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def build_for_mask_recovery_dynamic_span(self, batch_data_list, shared_basic_data):
        """
        mask value of dynamic multiple cells for each example, select one cell to predict --> recovery its value
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        bos_token_id = self.tokenizer_proxy.bos_token_id
        eos_token_id = self.tokenizer_proxy.eos_token_id
        pad_token_id = self.tokenizer_proxy.pad_token_id
        batch_row_features = []
        batch_prompt_prefix = []
        batch_prompt_value = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            fetched_result = fetch_dynamic_span(
                cur_row,
                self.dynamic_mask_missval_span,
                self.dynamic_mask_span_max_th,
                self.mask_token
            )
            if fetched_result is not None:
                batch_row_features.append(fetched_result[0])
                prompt_prefix = 'recovery {} :'.format(fetched_result[1])
                prompt_value = '{}'.format(fetched_result[2])
                batch_prompt_prefix.append(prompt_prefix)
                batch_prompt_value.append(prompt_value)
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_pprefix_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_prefix, self.max_prompt_tok_len)
        tmp_pvalue_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_value, self.max_feature_tok_len)
        batch_value_bos_ids, batch_value_eos_ids, batch_value_mask = post_process_tgt_value(
            tmp_pvalue_tok, bos_token_id, eos_token_id, pad_token_id
        )
        batch_prompt_prefix_ids = tmp_pprefix_tok['input_ids']
        batch_prompt_prefix_mask = tmp_pprefix_tok['attention_mask']
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        processed_data['batch_value_bos_ids'] = torch.LongTensor(batch_value_bos_ids)
        processed_data['batch_value_eos_ids'] = torch.LongTensor(batch_value_eos_ids)
        processed_data['batch_value_mask'] = torch.BoolTensor(batch_value_mask)
        return processed_data

    def build_for_mask_fill_missval_predict(self, batch_data_list, shared_basic_data):
        """
        common task:: mask and predict, acts as filling missing value.
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        common_fill_missval_column = self._common_fill_missval_column
        bos_token_id = self.tokenizer_proxy.bos_token_id
        eos_token_id = self.tokenizer_proxy.eos_token_id
        pad_token_id = self.tokenizer_proxy.pad_token_id
        batch_row_features = []
        batch_prompt_prefix = []
        batch_prompt_value = []
        batch_raw_data = []
        batch_raw_target = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            selected_col_value = cur_row[common_fill_missval_column]
            new_features = {cn: cur_row[cn] for cn in cur_row.keys()}
            new_features[common_fill_missval_column] = copy.deepcopy(selected_col_value)
            new_features[common_fill_missval_column]['processed_value'] = '{}'.format(self.mask_token)
            batch_row_features.append(new_features)
            prompt_prefix = 'recovery {} :'.format(common_fill_missval_column)
            prompt_value = '{}'.format(selected_col_value['processed_value'])
            batch_prompt_prefix.append(prompt_prefix)
            batch_prompt_value.append(prompt_value)
            if self.test_mode:
                batch_raw_data.append(item['raw_data'])
                batch_raw_target.append(selected_col_value['processed_value'])
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_pprefix_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_prefix, self.max_prompt_tok_len)
        tmp_pvalue_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_value, self.max_feature_tok_len)
        batch_value_bos_ids, batch_value_eos_ids, batch_value_mask = post_process_tgt_value(
            tmp_pvalue_tok, bos_token_id, eos_token_id, pad_token_id
        )
        batch_prompt_prefix_ids = tmp_pprefix_tok['input_ids']
        batch_prompt_prefix_mask = tmp_pprefix_tok['attention_mask']
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        processed_data['batch_value_bos_ids'] = torch.LongTensor(batch_value_bos_ids)
        processed_data['batch_value_eos_ids'] = torch.LongTensor(batch_value_eos_ids)
        processed_data['batch_value_mask'] = torch.BoolTensor(batch_value_mask)
        if self.test_mode:
            processed_data['batch_raw_data'] = batch_raw_data
            processed_data['batch_raw_target'] = batch_raw_target
        # info
        if self._gen_constrained_toks_ids is not None:
            processed_data['gen_infone_ids'] = torch.LongTensor(self._gen_constrained_toks_ids)
        return processed_data

    def build_for_recall(self, batch_data_list, shared_basic_data):
        """
        recall/memory the value of specific cell --> encourage the semantic representation to contains most of info
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        ori_cols = []
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            batch_row_features.append(cur_row)
            no_missing_col_names = list(cur_row.keys())  # allow cell value to be missing-value
            pidx = random.randint(0, len(no_missing_col_names) - 1)
            selected_col = no_missing_col_names[pidx]
            selected_col_value = cur_row[selected_col]
            ori_cols.append((selected_col, selected_col_value))

            prompt = 'memory {} : {} {}'.format(selected_col, selected_col_value['processed_value'], self.eos_token)
            prompt_prefix = 'memory {} :'.format(selected_col)
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        if len(batch_row_features) < 1:
            return None
        processed_data = {'basic_data': shared_basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def build_for_common_predict(self, batch_data_list, shared_basic_data):
        """
        recall/memory the value of specific cell --> encourage the semantic representation to contains most of info
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        test_tgt_anchor = ' '.join(['-'] * self.test_max_dec_steps)
        batch_prompt = []
        batch_prompt_prefix = []
        batch_raw_data = []
        batch_raw_target = []
        for bidx, item in enumerate(batch_data_list):
            tgt_result = item['target_label']
            if self.test_mode:
                batch_raw_data.append(item['raw_data'])
                batch_raw_target.append(item['target_label'])
                tgt_result = test_tgt_anchor
            prompt = self.common_pred_prefix + ' {} {}'.format(tgt_result, self.eos_token)
            prompt_prefix = self.common_pred_prefix
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        processed_data = {'basic_data': shared_basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        if self.test_mode:
            processed_data['batch_raw_data'] = batch_raw_data
            processed_data['batch_raw_target'] = batch_raw_target
        return processed_data

    def build_for_common_predict_sep_answer(self, batch_data_list, shared_basic_data):
        """
        recall/memory the value of specific cell --> encourage the semantic representation to contains most of info
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        const_prefix_tok = self.tokenizer_proxy.tokenize_to_numpy(self.common_pred_prefix, self.max_prompt_tok_len)
        prefix_ids_list = const_prefix_tok['input_ids'].tolist()[0]
        prefix_mask_list = [1] * len(prefix_ids_list)

        test_tgt_anchor = ' '.join(['-'] * self.test_max_dec_steps)
        if self.test_mode:
            batch_prompt_prefix_ids = [prefix_ids_list] * len(batch_data_list)
            batch_prompt_prefix_mask = [prefix_mask_list] * len(batch_data_list)
            batch_prompt_ids, batch_prompt_mask = batch_prompt_prefix_ids, batch_prompt_prefix_mask
            batch_raw_data = []
            batch_raw_target = []
            for bidx, item in enumerate(batch_data_list):
                batch_raw_data.append(item['raw_data'])
                batch_raw_target.append(item['target_label'])
        else:
            batch_prompt_value = []
            for bidx, item in enumerate(batch_data_list):
                tgt_result = item['target_label']
                batch_prompt_value.append(tgt_result)
            tmp_pvalue_tok = self.tokenizer_proxy.tokenize_to_numpy(batch_prompt_value, self.max_feature_tok_len)
            merge_result = merge_const_prompt_and_value(prefix_ids_list, tmp_pvalue_tok, self.eos_token_id)
            batch_prompt_ids, batch_prompt_mask, batch_prompt_prefix_ids, batch_prompt_prefix_mask = merge_result

        processed_data = {'basic_data': shared_basic_data}
        # to tensor
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        if self.test_mode:
            processed_data['batch_raw_data'] = batch_raw_data
            processed_data['batch_raw_target'] = batch_raw_target
        return processed_data

    def build_for_which_is_bigger(self, batch_data_list, shared_basic_data):
        """
        judge which column's value is bigger --> encourage the semantic representation to distinguish comparision
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            useful_col_names = [k for k, v in cur_row.items() if (not v['is_missing']) and 0 == v['ct_id']]
            if len(useful_col_names) < 2:
                continue
            random.shuffle(useful_col_names)
            col1, col2 = useful_col_names[0], useful_col_names[1]
            label = 'yes' if cur_row[col1]['ori_value'] > cur_row[col2]['ori_value'] else 'no'
            batch_row_features.append(cur_row)

            prompt = 'whether the value of {} is bigger than that of {} ? {}'.format(col1, col2, label)
            prompt_prefix = 'whether the value of {} is bigger than that of {} ?'.format(col1, col2)
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def build_for_predict_data_type(self, batch_data_list, shared_basic_data):
        ori_cols = []
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            batch_row_features.append(cur_row)
            no_missing_col_names = list(cur_row.keys())  # allow cell value to be missing-value
            pidx = random.randint(0, len(no_missing_col_names) - 1)
            selected_col = no_missing_col_names[pidx]
            selected_col_value = cur_row[selected_col]
            ori_cols.append((selected_col, selected_col_value))

            value = selected_col_value['dtype']
            prompt = 'what is the data type of {} : {}'.format(selected_col, value)
            prompt_prefix = 'what is the data type of {} :'.format(selected_col)
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        if len(batch_row_features) < 1:
            return None
        processed_data = {'basic_data': shared_basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def build_for_whether_is_missing_value(self, batch_data_list, shared_basic_data):
        """
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            column_names = list(cur_row.keys())
            no_missing_col_names = []
            missing_col_names = []
            for k,v in cur_row.items():
                if v['is_missing']:
                    missing_col_names.append(k)
                else:
                    no_missing_col_names.append(k)
            if random.random() <= 0.5:
                label = 'yes'
                if len(missing_col_names) > 0:
                    random.shuffle(missing_col_names)
                    missing_col = missing_col_names[0]
                    no_need_special_op = True
                else:
                    random.shuffle(no_missing_col_names)
                    missing_col = no_missing_col_names[0]
                    no_need_special_op = False
            else:
                label = 'no'
                no_need_special_op = True
                if len(no_missing_col_names) > 0:
                    random.shuffle(no_missing_col_names)
                    missing_col = no_missing_col_names[0]
                else:
                    label = 'yes'
                    random.shuffle(missing_col_names)
                    missing_col = missing_col_names[0]

            selected_col = missing_col
            selected_col_value = cur_row[selected_col]
            if no_need_special_op:
                batch_row_features.append(cur_row)
            else:
                new_features = {cn: cur_row[cn] for cn in column_names}
                new_features[selected_col] = copy.deepcopy(selected_col_value)
                new_features[selected_col]['processed_value'] = '{}'.format(self.missing_value_token)
                batch_row_features.append(new_features)
            prompt = 'whether column value is missing? {} : {}'.format(selected_col, label)
            prompt_prefix = 'whether column value is missing? {} :'.format(selected_col)
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def build_for_fill_missing_value(self, batch_data_list, shared_basic_data):
        """
        :param batch_data_list: List
        :param shared_basic_data: Dict
        :return:
        """
        ori_cols = []
        batch_row_features = []
        batch_prompt = []
        batch_prompt_prefix = []
        for bidx, item in enumerate(batch_data_list):
            cur_row = item['features']
            column_names = list(cur_row.keys())
            no_missing_col_names = [k for k, v in cur_row.items() if not v['is_missing']]
            if len(no_missing_col_names) < 1:
                continue
            pidx = random.randint(0, len(no_missing_col_names) - 1)
            selected_col = no_missing_col_names[pidx]
            selected_col_value = cur_row[selected_col]
            ori_cols.append((selected_col, selected_col_value))
            new_features = {cn: cur_row[cn] for cn in column_names}
            new_features[selected_col] = copy.deepcopy(selected_col_value)
            new_features[selected_col]['processed_value'] = '{}'.format(self.missing_value_token)
            batch_row_features.append(new_features)

            prompt = 'fill missing value of {} : {} {}'.format(selected_col, selected_col_value['processed_value'], self.eos_token)
            prompt_prefix = 'fill missing value of {} :'.format(selected_col)
            batch_prompt.append(prompt)
            batch_prompt_prefix.append(prompt_prefix)
        if len(batch_row_features) < 1:
            return None
        basic_data = wrap_subset_data(batch_row_features, self.tokenizer_proxy, self.max_feature_tok_len)
        processed_data = {'basic_data': basic_data}
        tmp_prompts = batch_prompt + batch_prompt_prefix
        tmp_prompts_tok = self.tokenizer_proxy.tokenize_to_numpy(tmp_prompts, self.max_prompt_tok_len)
        batch_size = len(batch_prompt)
        batch_prompt_ids = tmp_prompts_tok['input_ids'][: batch_size]
        batch_prompt_prefix_ids = tmp_prompts_tok['input_ids'][batch_size:]
        batch_prompt_mask = tmp_prompts_tok['attention_mask'][: batch_size]
        batch_prompt_prefix_mask = tmp_prompts_tok['attention_mask'][batch_size:]
        processed_data['batch_prompt_ids'] = torch.LongTensor(batch_prompt_ids)
        processed_data['batch_prompt_prefix_ids'] = torch.LongTensor(batch_prompt_prefix_ids)
        processed_data['batch_prompt_mask'] = torch.BoolTensor(batch_prompt_mask)
        processed_data['batch_prompt_prefix_mask'] = torch.BoolTensor(batch_prompt_prefix_mask)
        return processed_data

    def _trunc_features(self, batch_data_list):
        if self._filter_num_features:
            for item in batch_data_list:
                cur_row = item['features']
                if len(cur_row.keys()) > self.max_num_features:
                    column_names = [cn for cn in cur_row.keys()]
                    random.shuffle(column_names)
                    new_cns = column_names[:self.max_num_features]
                    new_cur_row = {cn: cur_row[cn] for cn in new_cns}
                    item['features'] = new_cur_row
        return batch_data_list

    def __call__(self, batch_data_list):
        """
        Take a list of subsets (views) from the original tests.
        """
        batch_data_list = self._trunc_features(batch_data_list)

        shared_basic_data = None
        if self.has_share_task:
            shared_basic_data = self._build_basic_data(batch_data_list)
        task_names = self.task_names
        result = {}
        for task in task_names:
            result[task] = self.task_register[task](batch_data_list, shared_basic_data)
        return result


def _build_positive_pairs(column_names, num_partition, overlap_ratio):
    """build multi-view of each sample by spliting columns
    """
    x_cols = column_names
    sub_col_list = np.array_split(np.array(x_cols), num_partition)
    len_cols = len(sub_col_list[0])
    overlap = int(math.ceil(len_cols * overlap_ratio))
    final_sub_cols = []
    for i, sub_col in enumerate(sub_col_list):
        if overlap > 0 and i < num_partition-1:
            sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
        elif overlap > 0 and i == num_partition-1:
            sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
        final_sub_cols.append(sub_col)
    return final_sub_cols


def _build_positive_pairs_single_view(column_names):
    x_cols = column_names
    final_sub_cols = [list(column_names)]
    n_corrupt = int(len(x_cols)*0.5)
    corrupt_cols = list(x_cols[:n_corrupt])
    np.random.shuffle(corrupt_cols)
    final_sub_cols.append(corrupt_cols)
    return final_sub_cols


def wrap_subset_data(batch_items, tokenizer_proxy, max_feature_tok_len=None):
    batch_names = []
    batch_values = []
    batch_types = []
    batch_row_col_nums = []
    batch_row_s_e = []
    for row in batch_items:
        start_idx = len(batch_types)
        for col_name in row.keys():
            col_val = row[col_name]
            batch_names.append(col_name)
            batch_values.append(col_val['processed_value'])
            batch_types.append(col_val['ct_id'])
        batch_row_col_nums.append(len(row.keys()))
        end_idx = len(batch_types)
        batch_row_s_e.append((start_idx, end_idx))
    batch_names_tok = tokenizer_proxy.tokenize_to_numpy(batch_names, max_feature_tok_len)
    batch_values_tok = tokenizer_proxy.tokenize_to_numpy(batch_values, max_feature_tok_len)
    # [N, max_len]
    batch_names_ids = batch_names_tok['input_ids']
    # [N, max_len]
    batch_names_mask = batch_names_tok['attention_mask']
    batch_values_ids = batch_values_tok['input_ids']
    batch_values_mask = batch_values_tok['attention_mask']
    # num of token for each column value
    batch_col_val_lens = []
    for mask in batch_values_mask:
        batch_col_val_lens.append(int(sum(mask)))
    batch_row_tok_nums = []
    for idx in range(len(batch_row_col_nums)):
        sidx, eidx = batch_row_s_e[idx]
        batch_row_tok_nums.append(int(sum(batch_col_val_lens[sidx:eidx]) + (eidx - sidx)))
    # wrap final result
    batch_max_len = int(max(batch_row_tok_nums))
    batch_need_pad_nums = []
    batch_final_mask = []
    for bidx in range(len(batch_row_col_nums)):
        batch_need_pad_nums.append(batch_max_len - batch_row_tok_nums[bidx])
        # first 1 means the position for CLS vector
        batch_final_mask.append(
            [1] * (1 + batch_row_tok_nums[bidx]) + [0] * batch_need_pad_nums[-1]
        )
    cn_ids = torch.LongTensor(batch_names_ids)
    cn_mask = torch.BoolTensor(batch_names_mask)
    cv_ids = torch.LongTensor(batch_values_ids)
    cv_mask = torch.BoolTensor(batch_values_mask)
    c_types = torch.LongTensor(batch_types)
    batch_final_mask = torch.BoolTensor(batch_final_mask)
    result = {
        'cn_ids': cn_ids,
        'cn_mask': cn_mask,
        'cv_ids': cv_ids,
        'cv_mask': cv_mask,
        'c_types': c_types,
        'batch_row_s_e': batch_row_s_e,
        'batch_max_len': batch_max_len,
        'batch_need_pad_nums': batch_need_pad_nums,
        'batch_final_mask': batch_final_mask,
    }
    return result


def merge_prompt_and_value(prefix_tok, value_tok, eos_token_id):
    max_len = prefix_tok['input_ids'].shape[1] + value_tok['input_ids'].shape[1]
    batch_prompt_ids = []
    batch_prompt_mask = []
    batch_size = prefix_tok['input_ids'].shape[0]
    batch_prefix_ids = prefix_tok['input_ids'].tolist()
    batch_prefix_mask = prefix_tok['attention_mask'].tolist()
    batch_value_ids = value_tok['input_ids'].tolist()
    batch_value_mask = value_tok['attention_mask'].tolist()
    new_max_len = 0
    for bidx in range(batch_size):
        cur_ori_ids = batch_prefix_ids[bidx] + batch_value_ids[bidx]
        cur_ori_mask = batch_prefix_mask[bidx] + batch_value_mask[bidx]
        cur_ids = []
        cur_mask = []
        for i in range(max_len):
            if 1 == cur_ori_mask[i]:
                cur_ids.append(cur_ori_ids[i])
                cur_mask.append(1)
        cur_ids.append(eos_token_id)
        cur_mask.append(1)
        batch_prompt_ids.append(cur_ids)
        batch_prompt_mask.append(cur_mask)
        if len(cur_ids) > new_max_len:
            new_max_len = len(cur_ids)
    batch_prompt_prefix_ids = []
    batch_prompt_prefix_mask = []
    prefix_pads = [0] * (new_max_len - len(batch_prefix_ids[0]))
    for bidx in range(batch_size):
        cur_prefix_ids = batch_prefix_ids[bidx]
        cur_prefix_mask = batch_prefix_mask[bidx]
        batch_prompt_prefix_ids.append(cur_prefix_ids + prefix_pads)
        batch_prompt_prefix_mask.append(cur_prefix_mask + prefix_pads)
        cur_ids, cur_mask = batch_prompt_ids[bidx], batch_prompt_mask[bidx]
        if len(cur_ids) < new_max_len:
            num_pad = new_max_len - len(cur_ids)
            pads = [0] * num_pad
            # cur_ids = cur_ids + pads
            # cur_mask = cur_mask + pads
            # batch_prompt_ids[bidx] = cur_ids[:new_max_len]
            # batch_prompt_mask[bidx] = cur_mask[:new_max_len]
            batch_prompt_ids[bidx] = cur_ids + pads
            batch_prompt_mask[bidx] = cur_mask + pads
    return batch_prompt_ids, batch_prompt_mask, batch_prompt_prefix_ids, batch_prompt_prefix_mask


def merge_const_prompt_and_value(prefix_ids_list, value_tok, eos_token_id):
    prefix_mask = [1] * len(prefix_ids_list)
    batch_value_ids = value_tok['input_ids'].tolist()
    batch_value_mask = value_tok['attention_mask'].tolist()
    pad_len = 1 + len(batch_value_ids[0])
    pad_ids = [0] * pad_len
    batch_size = len(batch_value_ids)
    batch_prompt_prefix_ids, batch_prompt_prefix_mask = [], []
    batch_prompt_ids = []
    batch_prompt_mask = []
    const_prefix_ids = prefix_ids_list + pad_ids
    const_prefix_mask = prefix_mask + pad_ids
    for bidx in range(batch_size):
        batch_prompt_prefix_ids.append(const_prefix_ids)
        batch_prompt_prefix_mask.append(const_prefix_mask)
        cur_prompt_ids = prefix_ids_list + batch_value_ids[bidx] + [0]
        cur_prompt_mask = prefix_mask + batch_value_mask[bidx] + [0]
        cur_len = sum(cur_prompt_mask)
        cur_prompt_ids[cur_len] = eos_token_id
        cur_prompt_mask[cur_len] = 1
        batch_prompt_ids.append(cur_prompt_ids)
        batch_prompt_mask.append(cur_prompt_mask)
    return batch_prompt_ids, batch_prompt_mask, batch_prompt_prefix_ids, batch_prompt_prefix_mask


def fetch_dynamic_span(
        item_features, dynamic_span_len_map,
        max_mask_th, missing_value_token):
    no_missing_col_names = []
    missing_col_names = []
    for cn, cv in item_features.items():
        if cv['is_missing']:
            missing_col_names.append(cn)
        else:
            no_missing_col_names.append(cn)
    if len(no_missing_col_names) < 1:
        return None
    new_item_features = copy.deepcopy(item_features)
    all_columns = [cn for cn in item_features.keys()]
    max_mask_num = int(len(all_columns) * max_mask_th)
    if max_mask_num < 1:
        max_mask_num = 1
    num_mask = 1
    if max_mask_num > 1:
        nrand = abs(np.random.normal())
        for cur_th, cur_len in dynamic_span_len_map:
            if nrand <= cur_th:
                num_mask = min(cur_len, max_mask_num)
                break
    random.shuffle(no_missing_col_names)
    selected_pred_cn = no_missing_col_names[0]
    selected_pred_cv = item_features[selected_pred_cn]['processed_value']
    new_item_features[selected_pred_cn]['processed_value'] = missing_value_token
    if num_mask > 1:
        remain_cns = no_missing_col_names[1:] + missing_col_names
        random.shuffle(remain_cns)
        for cidx in range(num_mask - 1):
            new_item_features[remain_cns[cidx]]['processed_value'] = missing_value_token
    return new_item_features, selected_pred_cn, selected_pred_cv


def post_process_tgt_value(value_tok, bos_token_id, eos_token_id, pad_token_id):
    max_len = value_tok['input_ids'].shape[1]
    batch_bos_value_ids = []
    batch_eos_value_ids = []
    batch_new_value_mask = []
    batch_size = value_tok['input_ids'].shape[0]
    batch_value_ids = value_tok['input_ids'].tolist()
    batch_value_mask = value_tok['attention_mask'].tolist()
    for bidx in range(batch_size):
        batch_bos_value_ids.append(
            [bos_token_id] + batch_value_ids[bidx]
        )
        batch_new_value_mask.append(
            [1] + batch_value_mask[bidx]
        )
        j = max_len - 1
        for i in range(max_len):
            j = max_len - 1 - i
            if pad_token_id != batch_value_ids[bidx][j]:
                break
        batch_eos_value_ids.append(
            batch_value_ids[bidx][:j + 1] + [eos_token_id] + batch_value_ids[bidx][j + 1:]
        )
    return batch_bos_value_ids, batch_eos_value_ids, batch_new_value_mask



