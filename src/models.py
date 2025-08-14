import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import einops

class TransformerEncoderNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        TransformerEncoder network

        Consisting of input embedding (linear projection),
        positional encoding, transformer encoder layers,
        and final prediction head (MLP)

        """
        super().__init__()
        self.emb = InputEmbeddingPosEncoding(input_dim=kwargs["input_dim"], d_model=kwargs["d_model"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=kwargs["d_model"],
            nhead=kwargs["nhead"],
            dim_feedforward=kwargs["dim_feedforward"],
            dropout=kwargs["dropout"],
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=kwargs["n_encoder_layers"],
        )
        self.prediction_head = nn.Linear(kwargs["d_model"], kwargs["output_dim"])

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        output = self.prediction_head(x)
        return output


class InputEmbeddingPosEncoding(nn.Module):
    def __init__(self, input_dim, d_model):
        """Combines input embedding and positional encoding"""
        super().__init__()
        self.lin_proj_layer = nn.Linear(
            in_features=input_dim,
            out_features=d_model
        )
        self.pos_encoder = AbsolutePositionalEncoding(d_model=d_model, dropout=0.0)

    def forward(self, x):
        x = self.lin_proj_layer(x)
        pe_x = self.pos_encoder(x)

        return x + pe_x


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000, batch_first=True):
        '''Sinusoidal positional encoding

        Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Parameters
        ----------
        d_model (int): input dimension
        dropout (float):  random zeroes input elements with probab. dropout
        max_len (int): max sequence length
        batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature). Default: True (batch, seq, feature).

        '''
        super().__init__()
        self.dropout_prob = dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = einops.rearrange(pe, 'S B D -> B S D')
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
                if batch_first else [batch_size, seq_len, embedding_dim]
        """
        if type(x) == list:
            x, _ = x  # Drop timestamp here
        if not self.batch_first:
            pe = self.pe[:x.size(0)]
        else:
            pe = self.pe[:, :x.size(1)]
        if self.dropout_prob != 0.0:
            return self.dropout(pe)
        else:
            return pe