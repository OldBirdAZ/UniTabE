# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import traceback
import math
from multiprocessing.pool import Pool
import random


data_description = """

The format of each line jsonl:

{
    "data": {
        "kv_s": {
            "column_1_name": "column_1_value",
            "column_2_name": "column_2_value",
            "column_3_name": 3.1415926
        },
        "target_label": -1,
        "num_features": ["column_3_name"],
        "bin_features": [],
        "example_id": 0,
    },
    "description": "dataset name"
}
"""


def wrap_each_sample(
        item, missing_value_token, retrieve_label_outof_fs=False,
        label_column_name='target_label', num_precision=4,
        target_label_mapping={}
):
    # TODO remove this assertion in the future
    assert missing_value_token == '[MASK]'
    processed_item = {}
    columns = item["data"]["kv_s"]
    num_features_clms = item["data"]["num_features"]
    bin_features_clms = item["data"]["bin_features"]
    if len(list(columns.keys())) < 1:
        return None

    for column_name, column_value in columns.items():
        processed_value = missing_value_token
        cur_column = {
            'ori_value': column_value,
        }
        processed_item[column_name] = cur_column

        if column_name in num_features_clms:
            if math.isnan(column_value):
                column_value = None
            cur_column["dtype"] = "number"
            cur_column['ct_id'] = 0
            if column_value is not None:
                cur_column['is_missing'] = False
                cv_1 = str(round(column_value, num_precision))
                cv_2 = list(cv_1)
                processed_value = ' '.join(cv_2)
            else:
                cur_column['is_missing'] = True
        elif column_name in bin_features_clms:
            if column_value is None or column_value == '':
                column_value = None
            cur_column["dtype"] = "binary"
            cur_column['ct_id'] = 1
            if column_value is not None and len(str(column_value).strip()) > 0:
                cur_column['is_missing'] = False
                processed_value = str(column_value).strip().lower()
            else:
                cur_column['is_missing'] = True
        else:
            if column_value is None or column_value == '':
                column_value = None
            cur_column["dtype"] = "string"
            cur_column['ct_id'] = 2
            if column_value is not None and len(str(column_value).strip()) > 0:
                cur_column['is_missing'] = False
                processed_value = str(column_value).strip().lower()
            else:
                cur_column['is_missing'] = True
        cur_column['processed_value'] = processed_value
    result = {
        'target_label': item['data']['target_label'],
        'features': processed_item
    }
    if retrieve_label_outof_fs:
        cur_label_column = processed_item[label_column_name]
        if cur_label_column['is_missing']:
            return None
        result['target_label'] = cur_label_column['processed_value']
        del processed_item[label_column_name]
    if len(target_label_mapping) > 0:
        tl_key = str(result['target_label'])
        if tl_key in target_label_mapping:
            result['target_label'] = target_label_mapping[tl_key]
        else:
            # raise ValueError('{} not in target_label_mapping'.format(tl_key))
            return None
    return result


class TabDataset(Dataset):
    def __init__(self, data_path, missing_value_token,
                 retrieve_label_outof_fs=False, label_column_name='target_label',
                 removed_features='', reader_num_processes=1, num_precision=4,
                 target_label_mapping='', max_num_examples=-1
                 ):
        super(TabDataset, self).__init__()
        self.missing_value_token = missing_value_token
        self.retrieve_label_outof_fs = retrieve_label_outof_fs
        self.label_column_name = label_column_name
        self.removed_features = removed_features.split(',')
        self.remove_f_flag = len(self.removed_features) > 0
        self.num_precision = num_precision
        self.target_label_mapping = {}
        target_label_mapping_pairs = [kv for kv in target_label_mapping.split('|') if len(kv) > 1 and ':' in kv]
        for kv in target_label_mapping_pairs:
            k, v = kv.split(':')
            self.target_label_mapping[k] = int(v)
        self.max_num_examples = max_num_examples
        self.trunc_with_max_num_examples = True if max_num_examples > 0 else False
        print('Loading examples from {}'.format(data_path))
        if reader_num_processes > 1:
            self._all_data = self._load_dataset_multi_processes(data_path, num_processes=reader_num_processes)
        else:
            self._all_data = self._load_dataset(data_path)
        self._num_examples = len(self._all_data)
        print('Loaded {} examples from {}'.format(self._num_examples, data_path))

    def _process_example(self, example_str):
        try:
            example = json.loads(example_str)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print('Error while load json example')
            return None
        item = None
        try:
            item = wrap_each_sample(
                example, self.missing_value_token,
                retrieve_label_outof_fs=self.retrieve_label_outof_fs,
                label_column_name=self.label_column_name,
                num_precision=self.num_precision,
                target_label_mapping=self.target_label_mapping
            )
            if item is not None:
                item['raw_data'] = example_str
                if self.remove_f_flag:
                    for cn in self.removed_features:
                        if cn in item['features']:
                            del item['features'][cn]
        except Exception as e:
            traceback.print_exc()
            print(e)
            # print(example)
            if 'description' in example:
                print('description: {}'.format(example['description']))
            if 'data' in example and 'example_id' in example['data']:
                print('example_id: {}'.format(example['data']['example_id']))
        return item

    def _load_dataset(self, data_path):
        raw_datas = []
        with open(data_path) as df:
            for line in df:
                example_str = line
                raw_datas.append(example_str)
        all_datas = []
        for example_str in raw_datas:
            item = self._process_example(example_str)
            if item is not None:
                all_datas.append(item)
        del raw_datas
        return all_datas

    def _load_dataset_multi_processes(self, data_path, num_processes=5):
        raw_datas = []
        with open(data_path) as df:
            for line in df:
                example_str = line
                raw_datas.append(example_str)
        if self.trunc_with_max_num_examples and self.max_num_examples <= len(raw_datas):
            print('Trunc Dataset to size: {}'.format(self.max_num_examples))
            select_ids = [data_idx for data_idx in range(len(raw_datas))]
            random.shuffle(select_ids)
            new_raw_datas = [raw_datas[data_idx] for data_idx in select_ids[:self.max_num_examples]]
            raw_datas = new_raw_datas

        cur_pool = Pool(num_processes)
        tmp_datas = cur_pool.map(self._process_example, raw_datas)
        cur_pool.close()
        cur_pool.join()

        all_datas = [item for item in tmp_datas if item is not None]
        del tmp_datas
        del raw_datas
        return all_datas

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        return self._all_data[index]


def build_data_loader(
        data_path, collate_fn, missing_value_token,
        batch_size, shuffle, num_workers=4, drop_last=True,
        retrieve_label_outof_fs=False, label_column_name='target',
        reader_num_processes=1, num_precision=4, target_label_mapping=''
    ):
    assert collate_fn is not None
    dataset = TabDataset(
        data_path, missing_value_token,
        retrieve_label_outof_fs=retrieve_label_outof_fs, label_column_name=label_column_name,
        reader_num_processes=reader_num_processes,
        num_precision=num_precision,
        target_label_mapping=target_label_mapping
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader



