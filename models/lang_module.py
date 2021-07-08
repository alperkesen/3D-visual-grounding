import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.selfattention import SelfAttention


class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
                 emb_size=300, hidden_size=256, use_self_attn=False):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.use_self_attn = use_self_attn

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size
        if self.use_self_attn:
            self.attention = SelfAttention(lang_size)

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        word_embs = data_dict["lang_feat"]
        input_lengths = data_dict["lang_len"]

        if self.use_self_attn:
            _, sorted_idx = torch.sort(input_lengths, descending=True)  # sort by length in descending order
            word_embs = word_embs[sorted_idx]
            lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
            # encode description
            feats, _ = self.gru(lang_feat)
            feats, _ = pad_packed_sequence(feats, batch_first=True)  # batch, timestep, hidden_size
            _, unsorted_idx = sorted_idx.sort()  # unsort in original order
            feats = feats[unsorted_idx]
            # self attention
            attn_weight = self.attention(feats) # batch, timestep, timestep
            _, T, _ = attn_weight.size()
            attn_value = torch.bmm(attn_weight, feats)  # B, T, H
            #lang_last,_ = torch.max(attn_value, 1)  # B, H
            lang_last = torch.sum(attn_value, dim=1) # B, H
            data_dict["attn_value"] = attn_value # B, T, H
            data_dict["attn_weight"] = attn_weight # B, T, T
            data_dict["attn_weight"] = torch.sum(data_dict["attn_weight"], dim=1) # B, T

        else:
            lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
            _, lang_last = self.gru(lang_feat)
            lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir

            _, sorted_idx = torch.sort(input_lengths, descending=True)  # sort by length in descending order
            word_embs = word_embs[sorted_idx]
            lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
            # encode description
            feats, _ = self.gru(lang_feat)
            feats, _ = pad_packed_sequence(feats, batch_first=True)  # batch, timestep, hidden_size
            _, unsorted_idx = sorted_idx.sort()  # unsort in original order
            feats = feats[unsorted_idx]
            data_dict["attn_value"] = feats

        # store the encoded language features
        data_dict["lang_emb"] = lang_last  # batch, hidden_size

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict
