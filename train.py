# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.trainer import TabTrainer
from unitab.opts import set_train_args
import streamtologger
import os


if __name__ == '__main__':
    config = set_train_args()
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    streamtologger.redirect(
        target=os.path.join(save_dir, 'log_train.txt'), append=True,
        header_format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] "
    )
    trainer = TabTrainer(config)
    train_mode = config.train_mode
    if 'default' == train_mode:
        trainer.train()
    elif 'sep_each_task' == train_mode:
        trainer.train_each_task()
    else:
        trainer.train()


