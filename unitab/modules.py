# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from torch import nn
import torch
import torch.nn.init as nn_init
import math
from unitab.transformer import TransformerEncoder, TransformerEncoderLayer


def _avg_embedding_by_mask(word_embs, att_mask=None):
    """
    :param word_embs: [N, x_len, emb_size]
    :param att_mask: [N, x_len]
    :return:
    """
    if att_mask is None:
        return word_embs.mean(1)
    else:
        word_embs.masked_fill_(att_mask.unsqueeze(2) == 0, 0.0)
        pat_embs = word_embs.sum(1) / att_mask.sum(1, keepdim=True)
        return pat_embs


class TabColumnEmb(nn.Module):
    def __init__(self, n_data_type, emb_size, hidden_size, word_emb, datatype_aware=False):
        super(TabColumnEmb, self).__init__()
        self.word_emb = word_emb
        self.datatype_aware = datatype_aware
        self.type_emb = nn.Embedding(n_data_type, emb_size)
        if self.datatype_aware:
            self.fuse_fn = nn.Sequential(
                nn.Linear(emb_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def forward(self, column_ids, column_ids_mask, datatype_ids):
        """
        :param column_ids: [N, x_len]
        :param column_ids_mask: [N, x_len]
        :param datatype_ids: [N]
        :return: [N, emb_size]
        """
        word_emb = self.word_emb(column_ids)
        pat_emb = _avg_embedding_by_mask(word_emb, column_ids_mask)
        if self.datatype_aware:
            # [N, emb_size]
            datatype_emb = self.type_emb(datatype_ids)
            gate = self.fuse_fn(datatype_emb)
            pat_emb = pat_emb + datatype_emb * gate
        return pat_emb


class TabColumnValueEmb(nn.Module):
    def __init__(self, word_emb):
        super(TabColumnValueEmb, self).__init__()
        self.word_emb = word_emb

    def forward(self, column_value_ids):
        """
        :param column_value_ids: [N, x_len]
        :return: [N, x_len, emb_size]
        """
        word_emb = self.word_emb(column_value_ids)
        return word_emb


class RelationLink(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(RelationLink, self).__init__()
        self.gate_fn = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, cn_emb, cv_emb):
        """
        :param cn_emb: [N, emb_size]
        :param cv_emb: [N, x_len, emb_size]
        :return: [N, x_len, emb_size]
        """
        gated_cn_emb = cn_emb * self.gate_fn(cn_emb)
        l_emb = cv_emb + gated_cn_emb.unsqueeze(1)
        return l_emb


class TabCLSEmb(nn.Module):
    """add a learnable cls token embedding at the end of each sequence.
    """
    def __init__(self, emb_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1, emb_size))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(emb_size), b=1/math.sqrt(emb_size))
        self.emb_size = emb_size

    def forward(self, batch_emb):
        """
        :param batch_emb: [N, max_num_columns, emb_dim]
        :return: [N, max_num_columns + 1, emb_dim]
        """
        return torch.cat([self.weight.repeat(batch_emb.size(0), 1, 1), batch_emb], 1)


class TabCellEmb(nn.Module):
    def __init__(self, n_data_type, emb_size, hidden_size, word_emb, datatype_aware=False):
        super(TabCellEmb, self).__init__()
        self.column_name_emb = TabColumnEmb(n_data_type, emb_size, hidden_size, word_emb, datatype_aware=datatype_aware)
        self.column_value_emb = TabColumnValueEmb(word_emb)
        self.column_cv_link = RelationLink(emb_size, hidden_size)
        self.cls_emb = TabCLSEmb(emb_size)
        pad_zeros = torch.zeros(1, emb_size)
        self.register_buffer('pad_zeros', pad_zeros)

    def forward(self, batch_data):
        """
        :param batch_data:
        :return: [N, max_num_columns + 1, emb_dim]
        """
        # [num_of_columns, emb_size]
        cn_emb = self.column_name_emb(batch_data['cn_ids'], batch_data['cn_mask'], batch_data['c_types'])
        # [num_of_columns, max_value_len, emb_size]
        cv_emb = self.column_value_emb(batch_data['cv_ids'])
        linked_cv_emb = self.column_cv_link(cn_emb, cv_emb)
        batch_row_s_e = batch_data['batch_row_s_e']
        batch_need_pad_nums = batch_data['batch_need_pad_nums']
        cv_mask = batch_data['cv_mask']
        batch_represent = []
        for bidx in range(len(batch_row_s_e)):
            item_s, item_e = batch_row_s_e[bidx]
            item_cn_emb = cn_emb[item_s: item_e].contiguous().view(-1, cn_emb.size(1))
            item_cv_emb = linked_cv_emb[item_s: item_e].contiguous().view(-1, linked_cv_emb.size(2))
            item_cv_mask = cv_mask[item_s: item_e].contiguous().view(-1)
            new_item_cv_emb = item_cv_emb[item_cv_mask]  # must be bool tensor!!
            # new_item_cv_emb = item_cv_emb.masked_select(item_cv_mask.unsqueeze(1)).view(-1, linked_cv_emb.size(2))
            if batch_need_pad_nums[bidx] > 0:
                new_item_cv_emb = torch.cat([new_item_cv_emb, self.pad_zeros.repeat(batch_need_pad_nums[bidx], 1)], 0)
            item_emb = torch.cat([item_cn_emb, new_item_cv_emb], 0)
            batch_represent.append(item_emb.unsqueeze(0))
        # [N, max_num_columns, emb_dim]
        batch_emb = torch.cat(batch_represent, 0)
        batch_emb = self.cls_emb(batch_emb)
        return batch_emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # [1, max_len, dim]
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb


class TabWordEmbedding(nn.Module):
    """
    word embedding with positional encoding
    """
    def __init__(self, vocab_size, emb_size, padding_idx=0, dropout=0.0, layer_norm_eps=1e-5):
        super(TabWordEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.position_embeddings = PositionalEncoding(dropout, dim=emb_size)
        self.norm = nn.LayerNorm(emb_size, eps=layer_norm_eps)

    def forward(self, x_ids):
        """
        :param x_ids: [N, x_max_len]
        :return:
        """
        embeddings = self.word_embeddings(x_ids)
        embeddings = self.position_embeddings(embeddings)
        embeddings = self.norm(embeddings)
        return embeddings


class TabEncoder(nn.Module):
    def __init__(self,
                 n_data_type, emb_size, vocab_size, pad_id,
                 hidden_size, n_head, ffn_size, n_enc_layer,
                 datatype_aware=False, dropout=0.1):
        super(TabEncoder, self).__init__()
        self.word_emb = TabWordEmbedding(vocab_size, emb_size, padding_idx=pad_id, dropout=dropout)
        self.tab_cell_emb = TabCellEmb(
            n_data_type=n_data_type, emb_size=emb_size,
            hidden_size=hidden_size, word_emb=self.word_emb,
            datatype_aware=datatype_aware
        )

        self.emb2enc = nn.Linear(emb_size, hidden_size)
        encoder_layer = TransformerEncoderLayer(
            hidden_size, nhead=n_head, dim_feedforward=ffn_size, dropout=dropout, batch_first=True
        )
        self.encoding_block = TransformerEncoder(encoder_layer, num_layers=n_enc_layer)

    def forward(self, batch_data):
        # [N, max_num_columns + 1, emb_dim]
        batch_emb = self.tab_cell_emb(batch_data)
        batch_states = self.emb2enc(batch_emb)
        batch_final_mask = batch_data['batch_final_mask']
        batch_pad_mask = ~batch_final_mask
        # [N, max_num_columns + 1, hidden_size]
        batch_enc = self.encoding_block(batch_states, src_key_padding_mask=batch_pad_mask)
        return batch_enc




