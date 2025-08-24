import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import einops

class TransformerEncoderNetwork(pl.LightningModule):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, dropout, n_encoder_layers, output_dim):
        """
        TransformerEncoder network for pre-training and embedding high frequency sensors

        Consisting of input embedding (linear projection),
        positional encoding, transformer encoder layers,
        and final prediction head (MLP)

        """
        super().__init__()
        self.emb = InputEmbeddingPosEncoding(input_dim=input_dim, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )
        self.prediction_head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        output = self.prediction_head(x)
        return output



class MaskedAutoencoder(pl.LightningModule):
    """
        A PyTorch Lightning module to handle the self-supervised pre-training of a
        TransformerEncoderNetwork using a masked autoencoder objective.
        """

    def __init__(self, model: TransformerEncoderNetwork, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['model'])


    def training_step(self, batch, batch_idx):
        masked_input: torch.Tensor = batch['masked_input']
        original_spectrograms: torch.Tensor = batch['original_spectrograms']
        binary_mask: torch.Tensor = batch['binary_mask']

        # The collate_fn gives us (batch, total_axes, freq_bins, time_frames).
        # The transformer needs (batch, seq_len, features).
        # Here, seq_len is time_frames, and features are total_axes * freq_bins.

        b, a, f, t = masked_input.shape

        # Reshape input for the model
        model_input = masked_input.permute(0, 3, 1, 2).reshape(b, t, a * f)

        reconstructed_output = self.model(model_input)

        # Reshape the output back to compare with the original spectrogram
        reconstructed_spectrograms = reconstructed_output.reshape(b, t, a, f).permute(0, 2, 3, 1)
        # print("\nRECONSTRUCTED")
        # print(f"  Shape: {reconstructed_spectrograms.shape}, Dtype: {reconstructed_spectrograms.dtype}")
        # print(f"  Mean: {reconstructed_spectrograms.mean():.4f}, Std: {reconstructed_spectrograms.std():.4f}")
        # print(f"  Min: {reconstructed_spectrograms.min():.4f}, Max: {reconstructed_spectrograms.max():.4f}")
        #
        # print("\nORIGINAL (Target):")
        # print(f"  Shape: {original_spectrograms.shape}, Dtype: {original_spectrograms.dtype}")
        # print(f"  Mean: {original_spectrograms.mean():.4f}, Std: {original_spectrograms.std():.4f}")
        # print(f"  Min: {original_spectrograms.min():.4f}, Max: {original_spectrograms.max():.4f}")

        loss_per_pixel = F.l1_loss(reconstructed_spectrograms, original_spectrograms, reduction='none')
        loss = (loss_per_pixel * binary_mask).sum() / (binary_mask.sum() + 1e-8)

        self.log('train_reconstruction_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Using AdamW optimizer as in the paper
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


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
        """Sinusoidal positional encoding

        Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Parameters
        ----------
        d_model (int): input dimension
        dropout (float):  random zeroes input elements with probab. dropout
        max_len (int): max sequence length
        batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature). Default: True (batch, seq, feature).

        """
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