# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.infer import TabInfer
from unitab.opts import set_train_args


if __name__ == '__main__':
    config = set_train_args()
    infer = TabInfer(config)
    infer.do_infer()




