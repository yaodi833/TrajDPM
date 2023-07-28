import copy
from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module
from torch.nn.init import xavier_uniform_


# import config
from traj_transformer.positional_encoding import PositionalEncoding
from traj_transformer.encoder import TransformerEncoderLayer, TransformerEncoder
from traj_transformer.decoder import TransformerDecoderLayer, TransformerDecoder


class TrajTransformer(Module):
    def __init__(
        self,
        pro_up_dim: int = 2,
        d_model: int = 512,
        pro_down_dim: int = 2,
        dim_feedforward: int = 2048,
        n_head: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(TrajTransformer, self).__init__()
        self.enc_embedding = Linear(pro_up_dim, d_model, bias=False)
        self.enc_pos_encode = PositionalEncoding(d_model=d_model)

        self.dec_embedding = Linear(pro_up_dim, d_model, bias=False)
        self.dec_pos_encode = PositionalEncoding(d_model=d_model)

        self.projection = Linear(d_model, pro_down_dim, bias=False)

        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, activation
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, n_head, dim_feedforward, dropout, activation
        )
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = n_head

    def forward(self, enc_tuple, dec_tuple):
        """
        # enc_inputs_tensor [batch_size, enc_max_len, fc_in_dim]
        # enc_padding_mask [batch_size, enc_max_len]
        # enc_traj_length [batch_size]
        """
        enc_inputs, enc_pad_mask = enc_tuple
        dec_inputs, dec_pad_mask = dec_tuple

        enc_inputs = torch.transpose(
            enc_inputs, 0, 1
        )  # [enc_max_len, batch_size, fc_in_dim]
        dec_inputs = torch.transpose(
            dec_inputs, 0, 1
        )  # [enc_max_len, batch_size, fc_in_dim]

        enc_inputs = self.enc_embedding(enc_inputs)
        enc_inputs = self.enc_pos_encode(enc_inputs)

        dec_inputs = self.dec_embedding(dec_inputs)
        dec_inputs = self.dec_pos_encode(dec_inputs)

        src = enc_inputs
        tgt = dec_inputs
        src_key_padding_mask = enc_pad_mask
        tgt_key_padding_mask = dec_pad_mask
        memory_key_padding_mask = enc_pad_mask

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        trans_outputs = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # trans_outputs [T, N, E]

        trans_outputs = torch.transpose(trans_outputs, 0, 1)
        # trans_outputs [N, T, E]

        outputs = self.projection(trans_outputs)
        # outputs [N, T, 2]

        return outputs

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# class TrajSimTrans(Module):
#     def __init__(
#         self,
#         pro_up_dim,
#         d_model,
#         pro_down_dim,
#         d_ff,
#         n_head,
#         num_encoder_layers,
#         num_decoder_layers,
#     ):
#         super(TrajSimTrans, self).__init__()
#         self.enc_embedding = Linear(pro_up_dim, d_model, bias=False)
#         self.enc_pos_encode = PositionalEncoding(d_model=d_model)

#         self.dec_embedding = Linear(pro_up_dim, d_model, bias=False)
#         self.dec_pos_encode = PositionalEncoding(d_model=d_model)

#         self.projection = Linear(d_model, pro_down_dim, bias=False)

#         self.transformer = Transformer(
#             d_model=d_model,
#             nhead=n_head,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=d_ff,
#         )

#         # self._reset_parameters()

#     def forward(self, enc_tuple, dec_tuple):
#         """
#         # enc_inputs_tensor [batch_size, enc_max_len, fc_in_dim]
#         # enc_padding_mask [batch_size, enc_max_len]
#         # enc_traj_length [batch_size]
#         """
#         enc_inputs, enc_pad_mask = enc_tuple
#         dec_inputs, dec_pad_mask = dec_tuple

#         enc_inputs = torch.transpose(
#             enc_inputs, 0, 1
#         )  # [enc_max_len, batch_size, fc_in_dim]
#         dec_inputs = torch.transpose(
#             dec_inputs, 0, 1
#         )  # [enc_max_len, batch_size, fc_in_dim]

#         enc_inputs = self.enc_embedding(enc_inputs)
#         enc_inputs = self.enc_pos_encode(enc_inputs)

#         dec_inputs = self.dec_embedding(dec_inputs)
#         dec_inputs = self.dec_pos_encode(dec_inputs)

#         trans_outputs = self.transformer(
#             src=enc_inputs,
#             tgt=dec_inputs,
#             src_key_padding_mask=enc_pad_mask,
#             tgt_key_padding_mask=dec_pad_mask,
#             memory_key_padding_mask=enc_pad_mask,
#         )
#         # trans_outputs [T, N, E]

#         trans_outputs = torch.transpose(trans_outputs, 0, 1)
#         # trans_outputs [N, T, E]

#         outputs = self.projection(trans_outputs)
#         # outputs [N, T, 2]

#         return outputs

#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer model."""

#         for p in self.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)
