
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import get_full_transformed_dataset, MaskingCollateFn, save_dataset_to_disk, load_dataset_from_disk
from models import TransformerEncoderNetwork, MaskedAutoencoder
from sensors import SENSOR_AXES_MAP, SensorFrequency, SENSOR_FREQUENCY_MAP, Sensor

if __name__ == '__main__':
    window_size = 250
    stride = 100
    seed = 42
    train_split_ratio = 0.8
    validation_split_ratio = 0.1
    n_fft = 50
    hop_length = 25
    win_length = 50
    time_mask_prob = 0.15
    freq_mask_prob = 0.20
    freq_bins = n_fft // 2 + 1
    batch_size = 64
    num_workers = 0
    d_model = 768
    nhead = 6
    dim_feedforward = 2048
    dropout = 0.1
    n_encoder_layers = 4
    max_epochs = 50

    reprocess_dataset = True

    if reprocess_dataset:
        dataset = get_full_transformed_dataset(train_split_ratio, validation_split_ratio, window_size, stride, seed, n_fft,
                                               hop_length, win_length)
        save_dataset_to_disk(dataset)

    dataset = load_dataset_from_disk()

    collate_fn = MaskingCollateFn(time_mask_prob=time_mask_prob, freq_mask_prob=freq_mask_prob)
    train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = True,
    )
    total_axes = sum(SENSOR_AXES_MAP[s] for s in Sensor if SENSOR_FREQUENCY_MAP[s] == SensorFrequency.HIGH)
    input_dim = total_axes * freq_bins

    transformer_model = TransformerEncoderNetwork(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        n_encoder_layers=n_encoder_layers,
        output_dim=input_dim
    )
    mae_trainer_module = MaskedAutoencoder(model=transformer_model)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(mae_trainer_module, train_loader)