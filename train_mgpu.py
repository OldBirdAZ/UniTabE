# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.opts import set_train_args
import streamtologger
import os
import torch
# from trainer_mgpu import build_model_and_tokenizer
from unitab.trainer_mgpu import build_model_and_tokenizer


if __name__ == '__main__':
    config = set_train_args()
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model, tokenizer = build_model_and_tokenizer(config)
    streamtologger.redirect(
        target=os.path.join(save_dir, 'config.txt'), append=True,
        header_format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] "
    )
    torch.save(config, os.path.join(save_dir, 'train_config.pt'))
    print(config)

    torch.save(model, os.path.join(save_dir, 'init-model-raw.pt'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'init-model-state.pt'))

    import subprocess
    # command_template = "WORK_DIR={} CUDA_VISIBLE_DEVICES={} python -m torch.distributed.launch --nproc_per_node={} trainer_mgpu.py"
    command_template = "PYTHONPATH=. WORK_DIR={} CUDA_VISIBLE_DEVICES={} python -m torch.distributed.launch --nproc_per_node={} ./unitab/trainer_mgpu.py"
    multi_gpu_ids = config.multi_gpu_ids
    num_gpus = len(multi_gpu_ids.split(','))
    cmd = command_template.format(save_dir, multi_gpu_ids, num_gpus)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()




