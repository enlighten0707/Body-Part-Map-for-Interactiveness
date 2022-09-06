# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from re import A
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Transformer_PartMap(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers_box=3, num_decoder_layers_verb=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, config= None):
        super().__init__()

        self.config = config
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer_box = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_box_norm = nn.LayerNorm(d_model)
        self.box_decoder = TransformerDecoder(decoder_layer_box, num_decoder_layers_box, decoder_box_norm, return_intermediate=return_intermediate_dec)

        decoder_layer_verb = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_verb_norm = nn.LayerNorm(d_model)

        self.binary_decoder_1 = TransformerDecoder(decoder_layer_verb, 1, decoder_verb_norm, return_intermediate=return_intermediate_dec)
        self.binary_decoder_2 = TransformerDecoder(decoder_layer_verb, 1, decoder_verb_norm, return_intermediate=return_intermediate_dec)
        self.binary_decoder_3 = TransformerDecoder(decoder_layer_verb, 1, decoder_verb_norm, return_intermediate=return_intermediate_dec)
        self.binary_decoder_4 = TransformerDecoder(decoder_layer_verb, 1, decoder_verb_norm, return_intermediate=return_intermediate_dec)
        self.classifier_6v = MLP(d_model, d_model, 12, 2)
        self.binary_weight = torch.nn.Parameter(torch.randn((6, d_model)), requires_grad=True)
        self.binary_bias = torch.nn.Parameter(torch.randn((6, d_model)), requires_grad=True)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers_verb = num_decoder_layers_verb

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_box(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        tgt_box  = torch.zeros_like(query_embed)
        out_box, box_decoder_weight  = self.box_decoder(tgt_box, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        
        return out_box.transpose(1, 2), memory # out_box (6=num_decoder, bz, 64=num_query, 256=dim)
    
    def forward_binary(self, memory, mask, pos_embed, out_box,
                        mask_part=None, mask_object=None, mask_human=None, num_queries=None, matched_6v=None):
        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        verb_query = out_box[-1] # (bz, num_query, dim)
        verb_query = verb_query.permute(1, 0, 2) # (num_query, bz, dim)
        tgt_verb   = torch.zeros_like(verb_query)
            
        out_binary, _ = self.binary_decoder_1(tgt_verb, 
                                            memory, pos=pos_embed, 
                                            query_pos=verb_query,
                                            memory_key_padding_mask=mask,
                                            ) # out_binary: (1, num_query, bz, dim), binary_decoder_weight: (1, bz, 64, H/32*W/32)
        ## mask matrix(bz, H/32*W/32)
        mask_part = mask_part.flatten(2, 3) # mask_part (bz, 6=num_part, H/32*W/32)
        mask_object = mask_object.flatten(2, 3) # mask_object (bz, num_query, H/32, W/32) -> (bz, num_query, H/32*W/32)
        mask_human = mask_human.flatten(2, 3)

        ## filter&merge
        score_part = self.classifier_6v(out_binary[-1].transpose(0, 1)) # (bz, num_query, dim) -> (bz, num_query, 12)
        score_part = score_part.view(*score_part.shape[:2], 6, 2) # (bz, num_query, 6, 2)
        score_part_sel = F.softmax(score_part, -1)[..., 1] # (bz, num_query, 6)
            
        _, index_max = score_part_sel.topk(3, dim=-1, largest = True) # get top_k part
        mask_select_part = []
        for bs in range(mask_part.shape[0]):
            mask_select_part.append(torch.stack([mask_part[bs][index_max[bs, :, j]] for j in range(index_max.shape[-1])], 0)) # (num_query, 6)
        mask_select_part = torch.stack(mask_select_part, 0)
        mask_part_dynamic = mask_select_part.min(dim=1)[0]
        # print("mask_part_dynamic",(~mask_part_dynamic).sum(-1)[0])
        mask_part_dynamic = torch.minimum(mask_part_dynamic, mask_object)

        part_queries = []
        for part_id in range(6):
            part_query = torch.einsum("ijk,k->ijk", verb_query, self.binary_weight[part_id]) + self.binary_bias[part_id] # (num_query, bz, dim), (dim) -> (num_query, bz, dim)
            part_queries.append(part_query)
        part_queries = torch.stack(part_queries, 0) # (6, num_query, bz, dim)
        score_part_softmax = score_part_sel
        part_query = torch.einsum("ijkl,kji->jkl", part_queries, score_part_softmax) # (num_query, bz, 256)
        part_query += verb_query


        ## progressively masking
        # the k-th body-part of the detected human, and the whole body of the other humans in the image
        mask_other_human = ~(mask_part.min(1)[0].unsqueeze(1).expand(-1, num_queries, -1) ^ mask_human)
        mask_2 = torch.minimum(mask_other_human, mask_part_dynamic)
        # print("mask_2",(~mask_2).sum(-1)[0])
        mask_2 = mask_2.unsqueeze(1).expand(-1, self.nhead, -1, -1).flatten(0, 1)
        out_binary, binary_decoder_weight_2 = self.binary_decoder_2(
                                            tgt_verb,
                                            memory, pos=pos_embed, 
                                            query_pos=part_query,
                                            memory_key_padding_mask=mask,
                                            memory_mask=mask_2,
                                            )
        # the k-th body-part from all persons in the image
        mask_3 = mask_part_dynamic
        # print("mask_3",(~mask_3).sum(-1)[0])
        mask_3 = mask_3.unsqueeze(1).expand(-1, self.nhead, -1, -1).flatten(0, 1) # (bz*num_heads, num_query, H/32*W/32)
        out_binary, binary_decoder_weight_3 = self.binary_decoder_3(out_binary[-1], 
                                            memory, pos=pos_embed, 
                                            query_pos=part_query,
                                            memory_key_padding_mask=mask,
                                            memory_mask=mask_3,
                                            )
        # the k-th body-part of the targeted person
        out_binary, binary_decoder_weight_4 = self.binary_decoder_4(out_binary[-1], 
                                            memory, pos=pos_embed, 
                                            query_pos=part_query,
                                            memory_key_padding_mask=mask,
                                            memory_mask=mask_3,
                                            )
        out_binary = out_binary.transpose(1, 2)

        return out_binary, score_part,\
            torch.stack([binary_decoder_weight_2, binary_decoder_weight_3, binary_decoder_weight_4], 0)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, depth=None, spmap=None):
        '''
            src:         (N, C, H, W)
            mask:        (N, H, W)
            query_embed: (Q, C)
            pos_embed:   (N, 128, H, W)
        '''
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) # (HW, N, C) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # (HW, N, C)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # (Q, N, C)
        mask = mask.flatten(1) # (N, HW)

        tgt = torch.zeros_like(query_embed) # (Q, N, C)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # (HW, N, C)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed) # (Q, N, C)
        return hs.transpose(1, 2), hs.transpose(1, 2), hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w) # (6, N, Q, C)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        '''
            src: (HW, N, C)
            mask: None
            src_key_padding_mask: (N, HW)
            pos: (HW, N, 128)
        '''
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
            tgt: (Q, N, C)
            memory: (HW, N, C)
            tgt_mask: None
            memory_mask: None
            tgt_key_padding_mask: None
            memory_key_padding_mask: (N, HW)
            pos: (HW, N, C)
            query_pos: (Q, N, C)
        '''
        output = tgt

        intermediate = []
        attn_weights = []

        for i, layer in enumerate(self.layers):
            output, attn_weight = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           ) # (Q, N, C)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                attn_weights.append(attn_weight)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(attn_weights) # (6, Q, N, C), (6, N, Q, S)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        '''
            src: (HW, N, C)
            src_mask: None
            src_key_padding_mask: (N, HW)
            pos: (HW, N, C)
        '''
        q = k = self.with_pos_embed(src, pos) # (HW, N, C)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] # (HW, N, C)
        src = src + self.dropout1(src2) # (HW, N, C)
        src = self.norm1(src) # (HW, N, C)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # (HW, N, C)
        src = src + self.dropout2(src2) # (HW, N, C)
        src = self.norm2(src) # (HW, N, C)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        '''
            tgt: (Q, N, C)
            memory: (HW, N, C)
            tgt_mask: None
            memory_mask: None
            tgt_key_padding_mask: None
            memory_key_padding_mask: (N, HW)
            pos: (HW, N, C)
            query_pos: (Q, N, C)
        '''
        q = k = self.with_pos_embed(tgt, query_pos) # (Q, N, C)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] # (Q, N, C)
        tgt = tgt + self.dropout1(tgt2) # (Q, N, C)
        tgt = self.norm1(tgt) # (Q, N, C)
        tgt2, attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask) # (Q, N, C)
        tgt = tgt + self.dropout2(tgt2) # (Q, N, C)
        tgt = self.norm2(tgt) # (Q, N, C)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # (Q, N, C)
        tgt = tgt + self.dropout3(tgt2) # (Q, N, C)
        tgt = self.norm3(tgt) # (Q, N, C)
        return tgt, attn_weight

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weight

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):

    return Transformer_PartMap(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers_box=args.dec_layers_box,
        num_decoder_layers_verb=args.dec_layers_verb,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        config = args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
