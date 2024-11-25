# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
from unitab.modules import TabEncoder
from unitab.train_utils import self_supervised_contrastive_loss_v2, self_supervised_contrastive_loss
EXTREME_SMALL = -1e30
from unitab import common_const
from torch.utils.checkpoint import checkpoint as gpu_efficient_ckpt
from unitab.loss_utils import FocalLoss


class ARDecAttn(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(ARDecAttn, self).__init__()
        self.y_proj_fn = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.x_proj_fn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.w_fn = nn.Linear(hidden_size, 1)

    def forward(self, y_prev_emb, y_prev_mask, enc_out, enc_out_mask):
        """
        :param y_prev_emb: [N, y_max_len, emb_size]
        :param y_prev_mask: [N, y_max_len]
        :param enc_out: [N, x_max_len, hidden_size]
        :param enc_out_mask: [N, x_max_len]
        :return: [N, hidden_size]
        """
        y = self.y_proj_fn(y_prev_emb)
        x = self.x_proj_fn(enc_out)
        # [N, y_max_len, x_max_len]
        y_attn_w = torch.matmul(y, x.transpose(1, 2))
        y_prev_mask_us = y_prev_mask.unsqueeze(2)
        attn_w_pad_mask = ~(y_prev_mask_us & enc_out_mask.unsqueeze(1))
        y_prev_pad_mask = ~y_prev_mask_us
        y_attn_w = y_attn_w.masked_fill(attn_w_pad_mask, float('-inf'))
        y_attn_prob = F.softmax(y_attn_w, 2)
        y_attn_prob = y_attn_prob.masked_fill(attn_w_pad_mask, 0.0)
        y_attn = torch.matmul(y_attn_prob, enc_out)

        w = self.w_fn(y_attn)
        w.masked_fill_(y_prev_pad_mask, float('-inf'))
        w = F.softmax(w, 1)
        w = w.masked_fill(y_prev_pad_mask, 0.0)
        attn_vec = torch.sum(w * y_attn, 1)
        return attn_vec


class ARDecoder(nn.Module):
    def __init__(self, word_embed, hidden_size, num_layers, dropout=0.0):
        super(ARDecoder, self).__init__()
        self.word_embed = word_embed
        assert isinstance(self.word_embed, nn.Embedding)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        emb_size = self.word_embed.weight.data.size()[-1]
        self.n_layer_state_fn = nn.Linear(hidden_size * 2, num_layers * 2 * hidden_size)
        self.ag_fn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fnn = nn.Linear(hidden_size, emb_size)
        self.attn_fn = ARDecAttn(emb_size, hidden_size)
        self.drop_fn = nn.Dropout(dropout)

    def init_state(self, y_prev_ids, y_prev_mask, enc_out, enc_out_mask):
        """
        :param y_prev_ids: [N, y_max_len]
        :param y_prev_mask: [N, y_max_len]
        :param enc_out: [N, x_max_len, hidden_size]
        :param enc_out_mask: [N, x_max_len]
        :return: 2 Tensor: [n_layer, N, hidden_size]
        """
        cls_vector = enc_out[:, 0]
        # [N, y_max_len, emb_size]
        y_prev_emb = self.word_embed(y_prev_ids)
        attn_vec = self.attn_fn(y_prev_emb, y_prev_mask, enc_out, enc_out_mask)
        vec = torch.cat([cls_vector, attn_vec], 1)

        states = self.n_layer_state_fn(vec).view(cls_vector.size(0), self.num_layers, self.hidden_size * 2)
        states = states.transpose(1, 0)
        return states[:, :, :self.hidden_size].contiguous(), states[:, :, self.hidden_size:].contiguous()

    def forward(self, y_prev_ids, y_prev_mask, y_ids, enc_out, enc_out_mask):
        """
        in-out format, e.g.:
            input:
                recovery column_name : column value
            output:
                column_name : column value
            here, ``recovery'' is used as the hint or prompt of the task

        :param y_prev_ids: [N, y_len]
        :param y_prev_mask: [N, y_len]
        :param y_ids: [N, y_max_len]
        :param enc_out: [N, x_max_len, hidden_size]
        :param enc_out_mask: [N, x_max_len]
        :return: [N, y_max_len, vocab_size]
        """
        init_state = self.init_state(y_prev_ids, y_prev_mask, enc_out, enc_out_mask)
        y_emb = self.word_embed(y_ids)
        y_emb = self.drop_fn(y_emb)
        y_hidden, (h_last, c_last) = self.ag_fn(y_emb, init_state)
        y_h = self.fnn(y_hidden)
        # [N, vocab_size]
        y_out = torch.matmul(y_h, self.word_embed.weight.transpose(0, 1))
        return y_out

    def generate(self, y_prev_ids, y_prev_mask, y_bos_ids, max_dec_step, enc_out, enc_out_mask, gen_infone_ids=None):
        """
        generate tokens step by step
        :param y_prev_ids: [N, y_len]
        :param y_prev_mask: [N, y_len]
        :param y_bos_ids: [N, 1]
        :param max_dec_step: int
        :param enc_out: [N, x_max_len, hidden_size]
        :param enc_out_mask: [N, x_max_len]
        :param gen_infone_ids: LongTensor, [max_num_infone_ids]
        :return: [N, max_dec_step, vocab_size]
        """
        assert max_dec_step > 0
        assert y_bos_ids.size(1) == 1
        init_state = self.init_state(y_prev_ids, y_prev_mask, enc_out, enc_out_mask)
        y_emb = self.word_embed(y_bos_ids)
        y_hidden, (h_last, c_last) = self.ag_fn(y_emb, init_state)
        y_h_last_step = self.fnn(y_hidden)
        # [N, 1, vocab_size]
        y_out = torch.matmul(y_h_last_step, self.word_embed.weight.transpose(0, 1))
        # [1, 1, vocab_size]
        gen_infone_mask = None
        if gen_infone_ids is not None:
            gen_infone_ids = gen_infone_ids.unsqueeze(0).unsqueeze(0)
            gen_infone_mask = torch.ones_like(y_out[:1, :1, :]) > 0
            gen_infone_mask = torch.scatter(gen_infone_mask, 2, gen_infone_ids, False)
        if gen_infone_mask is not None:
            y_out = torch.masked_fill(y_out, gen_infone_mask, float('-inf'))
        y_prev_ids = torch.argmax(y_out, -1)  # [N, 1]

        dec_outputs = []
        dec_outputs_ids = []
        dec_outputs.append(y_out)
        dec_outputs_ids.append(y_prev_ids)
        for step in range(max_dec_step - 1):
            y_emb = self.word_embed(y_prev_ids)
            y_hidden, (h_last, c_last) = self.ag_fn(y_emb, (h_last, c_last))
            y_h = self.fnn(y_hidden)
            y_out = torch.matmul(y_h, self.word_embed.weight.transpose(0, 1))
            if gen_infone_mask is not None:
                y_out = torch.masked_fill(y_out, gen_infone_mask, float('-inf'))
            y_prev_ids = torch.argmax(y_out[:, -1:], -1)  # [N, 1]
            dec_outputs.append(y_out)
            dec_outputs_ids.append(y_prev_ids)
        result = {
            'dec_outputs': torch.cat(dec_outputs, 1),
            'dec_outputs_ids': torch.cat(dec_outputs_ids, 1),
            'useful_gen_infone_mask': ~gen_infone_mask if gen_infone_mask is not None else None
        }
        return result


class TabModel(nn.Module):
    def __init__(self, n_data_type, emb_size, vocab_size, pad_id, hidden_size,
                 n_head, ffn_size, n_enc_layer, n_dec_layer,
                 datatype_aware=False, dropout=0.0, temperature=1.0, use_memory_efficient=False):
        super(TabModel, self).__init__()
        self.tab_encoder = TabEncoder(
            n_data_type=n_data_type,
            emb_size=emb_size,
            vocab_size=vocab_size,
            pad_id=pad_id,
            hidden_size=hidden_size,
            n_head=n_head,
            ffn_size=ffn_size,
            n_enc_layer=n_enc_layer,
            dropout=dropout,
            datatype_aware=datatype_aware
        )
        pure_word_emb = self.tab_encoder.word_emb.word_embeddings
        self.tab_decoder = ARDecoder(
            word_embed=pure_word_emb,
            hidden_size=hidden_size,
            num_layers=n_dec_layer,
            dropout=dropout
        )

        # train loss function
        self.classification_loss_fn = nn.NLLLoss()
        self.register_classification_loss_fn()
        self.dec_temperature = temperature
        self.use_memory_efficient = use_memory_efficient

        self.task_register = {
            common_const.TASK_NAME_CONTRASTIVE_LEARN: self.train_contrastive_learning_with_dec,
            common_const.TASK_NAME_CONTRASTIVE_LEARN2: self.train_contrastive_learning_with_cls,
            common_const.TASK_NAME_MASK_RECOVERY: self.train_common,
            common_const.TASK_NAME_DYNAMIC_MASK_SPAN_RECOVERY: self.train_common,
            common_const.TASK_NAME_RECALL: self.train_common,
            common_const.TASK_NAME_CLASSIFICATION: self.train_classification2,
            common_const.TASK_NAME_COMMON_PREDICTION: self.train_common,
            common_const.TASK_NAME_COMMON_PREDICTION_SEP_ANSWER: self.train_common,
            common_const.TASK_NAME_WHICH_IS_BIGGER: self.train_common,
            common_const.TASK_NAME_DATA_TYPE: self.train_common,
            common_const.TASK_NAME_WHETHER_IS_MISSING_VALUE: self.train_common,
            common_const.TASK_NAME_FILL_MISSING_VALUE: self.train_common,
        }
        self.token_mode_loss_fn = nn.NLLLoss(ignore_index=pad_id, reduction='mean')
        dec_out_pad_tensor = torch.zeros([1, 1, vocab_size])
        self.register_buffer('dec_out_pad_tensor', dec_out_pad_tensor)
        vocab_seq = torch.LongTensor([xidx for xidx in range(vocab_size)]).unsqueeze(0)
        self.register_buffer('vocab_seq', vocab_seq)

    def call_encoder(self, basic_data):
        if self.use_memory_efficient:
            return gpu_efficient_ckpt(self.tab_encoder, basic_data)
        return self.tab_encoder(basic_data)

    def call_decoder(self, y_prev_ids, y_prev_mask, y_ids, enc_out, enc_out_mask):
        # if self.use_memory_efficient:
        #     dec_out = gpu_efficient_ckpt(self.tab_decoder, y_prev_ids, y_prev_mask, y_ids, enc_out, enc_out_mask)
        # else:
        #     dec_out = self.tab_decoder(
        #         y_prev_ids=y_prev_ids,
        #         y_prev_mask=y_prev_mask,
        #         y_ids=y_ids,
        #         enc_out=enc_out,
        #         enc_out_mask=enc_out_mask
        #     )
        dec_out = self.tab_decoder(
            y_prev_ids=y_prev_ids,
            y_prev_mask=y_prev_mask,
            y_ids=y_ids,
            enc_out=enc_out,
            enc_out_mask=enc_out_mask
        )
        return dec_out

    def register_classification_loss_fn(self):
        self.cal_classification_fn = self.calculate_cls_loss

    def calculate_cls_focal_loss(self, raw_logits, targets):
        return self.classification_focal_loss_fn(raw_logits, targets)
    def register_classification_focal_loss_fn(self, gamma):
        self.classification_focal_loss_fn = FocalLoss(
            gamma=gamma
        )
        self.cal_classification_fn = self.calculate_cls_focal_loss

    def train_contrastive_learning_with_cls(self, wrapped_data):
        p1_data, p2_data = wrapped_data
        p1_basic_data = p1_data['basic_data']
        p2_basic_data = p2_data['basic_data']

        p1_states = self.call_encoder(p1_basic_data)
        p1_vector = p1_states[:, 0]

        p2_states = self.call_encoder(p2_basic_data)
        p2_vector = p2_states[:, 0]

        # neg_pair_ids = p1_data['neg_pair_ids']
        # loss, acc = self_supervised_contrastive_loss_v2(neg_pair_ids, p1_vector, p2_vector)
        pairs = torch.cat([p1_vector.unsqueeze(1), p2_vector.unsqueeze(1)], 1)
        loss, acc = self_supervised_contrastive_loss(pairs)

        result = {
            'loss': loss,
            'log_vals': {
                'loss': loss.item(),
                'acc': acc,
            }
        }
        return result

    def train_contrastive_learning_with_dec(self, wrapped_data):
        p1_data, p2_data = wrapped_data
        p1_basic_data = p1_data['basic_data']
        p2_basic_data = p2_data['basic_data']
        p1_batch_final_mask = p1_basic_data['batch_final_mask']
        p2_batch_final_mask = p2_basic_data['batch_final_mask']

        p1_states = self.call_encoder(p1_basic_data)
        p1_dec_out = self.call_decoder(
            y_prev_ids=p1_data['batch_prompt_ids'],
            y_prev_mask=p1_data['batch_prompt_mask'],
            y_ids=p1_data['batch_prompt_ids'],
            enc_out=p1_states,
            enc_out_mask=p1_batch_final_mask
        )
        p1_vector = p1_dec_out[:, -1]

        p2_states = self.call_encoder(p2_basic_data)
        p2_dec_out = self.call_decoder(
            y_prev_ids=p1_data['batch_prompt_ids'],
            y_prev_mask=p1_data['batch_prompt_mask'],
            y_ids=p1_data['batch_prompt_ids'],
            enc_out=p2_states,
            enc_out_mask=p2_batch_final_mask
        )
        p2_vector = p2_dec_out[:, -1]

        neg_pair_ids = p1_data['neg_pair_ids']
        loss, acc = self_supervised_contrastive_loss_v2(neg_pair_ids, p1_vector, p2_vector)
        # pairs = torch.cat([p1_vector.unsqueeze(1), p2_vector.unsqueeze(1)], 1)
        # loss, acc = self_supervised_contrastive_loss(pairs)

        result = {
            'loss': loss,
            'log_vals': {
                'loss': loss.item(),
                'acc': acc,
            }
        }
        return result

    def train_common(self, wrapped_data):
        basic_data = wrapped_data['basic_data']
        batch_final_mask = basic_data['batch_final_mask']
        # states = self.tab_encoder(basic_data)
        states = self.call_encoder(basic_data)
        batch_prompt_prefix_ids = wrapped_data['batch_prompt_prefix_ids']
        batch_prompt_prefix_mask = wrapped_data['batch_prompt_prefix_mask']
        batch_value_bos_ids = wrapped_data['batch_value_bos_ids']
        batch_value_eos_ids = wrapped_data['batch_value_eos_ids']
        batch_value_mask = wrapped_data['batch_value_mask']
        gen_infone_ids = None
        if 'gen_infone_ids' in wrapped_data:
            gen_infone_ids = wrapped_data['gen_infone_ids']
        dec_out = self.call_decoder(
            y_prev_ids=batch_prompt_prefix_ids,
            y_prev_mask=batch_prompt_prefix_mask,
            y_ids=batch_value_bos_ids,
            enc_out=states,
            enc_out_mask=batch_final_mask
        )
        dec_out_valid = dec_out.view(-1, dec_out.size(2))
        dec_tgt_ids = batch_value_eos_ids.view(-1)
        batch_size = dec_out.size(0)
        classification_probs = None
        if 'infone_ids' in wrapped_data and wrapped_data['infone_ids'] is not None:
            # currently, only used by classification tasks
            # [1, num labels]
            infone_ids = wrapped_data['infone_ids']
            # infone_to_mask = self.vocab_seq.scatter(1, infone_ids, -100) != -100
            # dec_out_valid = dec_out_valid.masked_fill(infone_to_mask, EXTREME_SMALL)
            # classification_logits = dec_out_valid.gather(1, infone_ids.repeat(dec_out_valid.size(0), 1))
            # classification_probs = F.softmax(classification_logits, -1)
            # batch_target_label = wrapped_data['batch_target_label']
            # loss = self.classification_loss_fn(F.log_softmax(classification_logits, -1), batch_target_label)

            # [N, num labels]
            classification_logits = torch.index_select(dec_out_valid, 1, infone_ids.squeeze(0))
            classification_probs = F.softmax(classification_logits, -1)
            batch_target_label = wrapped_data['batch_target_label']
            # loss = self.classification_loss_fn(F.log_softmax(classification_logits, -1), batch_target_label)
            loss = self.cal_classification_fn(classification_logits, batch_target_label)
            tok_acc = torch.mean((torch.argmax(classification_logits, 1) == batch_target_label).float()).item()
        elif gen_infone_ids is not None:
            # [1, num_gen_infone]
            gen_infone_ids = gen_infone_ids.unsqueeze(0)
            gen_infone_mask = torch.ones_like(dec_out_valid[:1, :]) > 0
            gen_infone_mask = torch.scatter(gen_infone_mask, 1, gen_infone_ids, False)
            dec_out_valid = torch.masked_fill(dec_out_valid, gen_infone_mask, float('-inf'))
            loss = self.token_mode_loss_fn(F.log_softmax(dec_out_valid / self.dec_temperature, -1), dec_tgt_ids)
            tok_acc = torch.mean((torch.argmax(dec_out_valid, 1) == dec_tgt_ids).float()).item()
        else:
            loss = self.token_mode_loss_fn(F.log_softmax(dec_out_valid / self.dec_temperature, -1), dec_tgt_ids)
            tok_acc = torch.mean((torch.argmax(dec_out_valid, 1) == dec_tgt_ids).float()).item()
        result = {
            'loss': loss,
            'log_vals': {
                'loss': loss.item(),
                'acc': tok_acc,
                'num_dec_out_toks': torch.sum(batch_value_mask).item() / batch_size
            },
            'classification_probs': classification_probs,
            'dec_out_valid': dec_out_valid,
            'dec_out': dec_out,
            'batch_size': batch_size,
        }
        return result

    def infer_common(self, wrapped_data, max_dec_steps):
        basic_data = wrapped_data['basic_data']
        batch_final_mask = basic_data['batch_final_mask']
        # states = self.tab_encoder(basic_data)
        states = self.call_encoder(basic_data)
        batch_prompt_prefix_ids = wrapped_data['batch_prompt_prefix_ids']
        batch_prompt_prefix_mask = wrapped_data['batch_prompt_prefix_mask']
        batch_value_bos_ids = wrapped_data['batch_value_bos_ids']
        batch_value_eos_ids = wrapped_data['batch_value_eos_ids']
        gen_infone_ids = None
        if 'gen_infone_ids' in wrapped_data:
            gen_infone_ids = wrapped_data['gen_infone_ids']

        batch_size = batch_prompt_prefix_ids.size(0)
        # dec_outputs, dec_outputs_ids = self.tab_decoder.generate(
        dec_results = self.tab_decoder.generate(
            y_prev_ids=batch_prompt_prefix_ids,
            y_prev_mask=batch_prompt_prefix_mask,
            y_bos_ids=batch_value_bos_ids[:, :1],
            max_dec_step=max_dec_steps,
            enc_out=states,
            enc_out_mask=batch_final_mask,
            gen_infone_ids=gen_infone_ids
        )
        dec_outputs, dec_outputs_ids = dec_results['dec_outputs'], dec_results['dec_outputs_ids']
        useful_gen_infone_mask = dec_results['useful_gen_infone_mask']

        result = {
            'dec_out_valid': dec_outputs,
            'dec_out': dec_outputs,
            'dec_out_ids': dec_outputs_ids,
            'batch_size': batch_size,
            'gen_infoned_out': None,
            'gen_infoned_probs': None,
        }
        if gen_infone_ids is not None:
            # [N, dec_y_len, num_infoned_toks]
            infoned_out = torch.index_select(dec_outputs, 2, gen_infone_ids)
            infoned_probs = torch.softmax(infoned_out, -1)
            result['gen_infoned_out'] = infoned_out
            result['gen_infoned_probs'] = infoned_probs
        return result

    def train_classification2(self, wrapped_data):
        return self.train_common(wrapped_data)

    def export_representation(self, wrapped_data):
        basic_data = wrapped_data['basic_data']
        batch_final_mask = basic_data['batch_final_mask']
        states = self.call_encoder(basic_data)
        result = {
            'states': states,
            'batch_final_mask': batch_final_mask
        }
        return result

    def fetch_register_task(self, task_name):
        if task_name in self.task_register:
            return self.task_register[task_name]
        return self.train_common

    def forward(self, wrapped_data, task_name):

        """
        :param wrapped_data:
        :param task_name:
        :return:
        """
        return self.fetch_register_task(task_name)(wrapped_data)







