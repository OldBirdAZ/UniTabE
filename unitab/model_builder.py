# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from unitab.models import TabModel
from unitab.tab_data_process import TabularPretrainCollator


def build_pretrain_model(model_config, pad_id, vocab_size):
    n_data_type = model_config.n_data_type
    emb_size = model_config.emb_size
    hidden_size = model_config.hidden_size
    n_head = model_config.n_head
    ffn_size = model_config.ffn_size
    n_enc_layer = model_config.n_enc_layer
    n_dec_layer = model_config.n_dec_layer
    dropout = model_config.dropout

    model = TabModel(
        n_data_type=n_data_type,
        emb_size=emb_size,
        vocab_size=vocab_size,
        pad_id=pad_id,
        hidden_size=hidden_size,
        n_head=n_head,
        ffn_size=ffn_size,
        n_enc_layer=n_enc_layer,
        n_dec_layer=n_dec_layer,
        dropout=dropout,
        temperature=model_config.temperature,
        datatype_aware=model_config.datatype_aware,
        use_memory_efficient=model_config.use_memory_efficient
    )
    return model


def build_data_collator(config, tokenizer_proxy, pretrain_task_names, verbose=True):
    batch_data_collate_fn = TabularPretrainCollator(
        tokenizer_proxy,
        task_names=pretrain_task_names,
        overlap_ratio=config.overlap_ratio,
        num_partition=config.num_partition,
        num_classification_labels=config.num_classification_labels,
        max_num_features=config.max_num_features,
        max_feature_tok_len=config.max_feature_tok_len,
        max_prompt_tok_len=config.max_prompt_tok_len,
        common_pred_prefix=config.common_pred_prefix,
        test_mode=config.test_mode,
        test_max_dec_steps=config.test_max_dec_steps,
        verbose=verbose,
        data_config=config
    )

    return batch_data_collate_fn
