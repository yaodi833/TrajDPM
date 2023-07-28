import torch
from torch.nn import Module, Linear, LayerNorm, BatchNorm1d

from traj_transformer.positional_encoding import PositionalEncoding
from traj_transformer.encoder import TransformerEncoderLayer, TransformerEncoder
from torch.nn.init import xavier_uniform_


class TrajEmbedding(Module):
    def __init__(self, fc_up_in_dim, d_model, d_ff, n_head, num_encoder_layers, max_length, device):
        super(TrajEmbedding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.enc_embedding = Linear(fc_up_in_dim, d_model, bias=False)
        self.enc_pos_encode = PositionalEncoding(d_model=d_model, max_len=max_length)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.output_norm = BatchNorm1d(d_model)
        self._reset_parameters()

    def forward(self, inputs, this_mask, this_lengths):
        """
        inputs [batch_size, sample_num, sequence_len, embedding_size]
        outputs [batch_size, sample_num, 10]
        """
        src_key_padding_mask = this_mask
        traj_lengths = this_lengths

        batch_num = inputs.size(0)
        sample_num = inputs.size(1)
        max_traj_len = inputs.size(2)
        fc_in_dim = inputs.size(3)

        inputs = inputs.reshape((batch_num * sample_num, max_traj_len, fc_in_dim))
        # inputs [batch_size * sample_num, sequence_len, fc_in_dim]

        inputs = torch.transpose(inputs, 0, 1)
        # inputs [sequence_len, batch_size * sample_num, fc_in_dim]

        inputs = self.enc_embedding(inputs)
        inputs = self.enc_pos_encode(inputs)
        # inputs [sequence_len, batch_size * sample_num, embedding_size]
        enc_out = self.encoder(inputs, src_key_padding_mask=src_key_padding_mask)
        # enc_out [sequence_len, batch_size * sample_num, embedding_size]

        enc_out = torch.transpose(enc_out, 0, 1)
        # enc_out [batch_size * sample_num, sequence_len, embedding_size]

        out = torch.zeros((batch_num * sample_num, self.d_model)).to(self.device)

        for traj_index, this_len in enumerate(traj_lengths):
            this_vec = torch.sum(enc_out[traj_index][:this_len], dim=0) / this_len
            out[traj_index] = this_vec
        out = self.output_norm(out)
        out = out.reshape(batch_num, sample_num, self.d_model)
        # [batch_size, sample_num, d_model]
        
        return out

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()
