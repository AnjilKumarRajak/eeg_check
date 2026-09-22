"""VERBATIM port of the base paper's model architecture.

Source: Wang & Ji, "Open Vocabulary Electroencephalography-to-Text Decoding and
Zero-Shot Sentiment Classification" (AAAI 2022), official repository
`model_decoding.py` (class BrainTranslator). Copied unchanged except for the
removal of unused imports/commented-out debug lines and one generation-time
compatibility note (see WangJiOfficial in experiments/e6_seq2seq_baseline.py —
their generate() forwards `labels` into HuggingFace generate(), which newer
transformers versions reject; the adapter calls pretrained.generate() without
that kwarg, which is computationally identical).

This module exists so the 1:1 baseline row uses THEIR model code, with only the
data pipeline (our leakage-free preprocessing) swapped underneath it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

""" main architecture for open vocabulary EEG-To-Text decoding"""


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()

        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer,
                                                        num_layers=6)
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = self.additional_encoder(
            input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(self, input_embeddings_batch, input_masks_batch, input_masks_invert,
                 target_ids_batch_converted, **kwargs):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            labels=target_ids_batch_converted,
            return_dict=True,
            **kwargs)
        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert,
                target_ids_batch_converted):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        out = self.pretrained(inputs_embeds=encoded_embedding,
                              attention_mask=input_masks_batch,
                              return_dict=True, labels=target_ids_batch_converted)
        return out


def apply_step1_freeze(model: nn.Module) -> list[str]:
    """Their step-1 freeze rule, verbatim (train_decoding.py): freeze every pretrained
    BART parameter EXCEPT shared embeddings, positional embeddings, and encoder layer 0.
    Matches both `pretrained.*` (their class) and `backbone.*` (our reimplementation)
    naming. Returns the names left trainable, for the sanity print."""
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("pretrained" in name or "backbone" in name):
            if ("shared" in name) or ("embed_positions" in name) or ("encoder.layers.0" in name):
                trainable.append(name)
                continue
            param.requires_grad = False
        elif param.requires_grad:
            trainable.append(name)
    return trainable
