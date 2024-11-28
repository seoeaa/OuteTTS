from dataclasses import dataclass
import logging
import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)
logging.basicConfig(level=logging.WARNING)  # Изменили уровень логирования на WARNING
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


def normalize_audio(waveform: torch.Tensor, target_db: float = -3.0) -> torch.Tensor:
    """
    Нормализует аудио к целевому уровню в децибелах
    """
    # Находим максимальную амплитуду
    max_amp = torch.max(torch.abs(waveform))
    if max_amp == 0:
        return waveform
    
    # Вычисляем текущий уровень в dB
    current_db = 20 * torch.log10(max_amp)
    
    # Вычисляем необходимый коэффициент усиления
    gain = 10 ** ((target_db - current_db) / 20)
    
    # Применяем усиление
    return waveform * gain


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        logger.info(f"Initialized dataset with {len(self.filelist)} files")

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_info = self.filelist[index]
        audio_path = file_info.split('|')[0]  # Split by | to handle format "path|text"
        
        try:
            y, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {str(e)}")
            y = torch.zeros((1, self.num_samples))
            sr = self.sampling_rate
        
        if y.ndim > 2:
            y = y.mean(dim=-1, keepdim=False)
            
        # Нормализация с случайным целевым уровнем для обучения
        target_db = np.random.uniform(-6, -1) if self.train else -3
        y = normalize_audio(y, target_db)
        
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]

        return y[0]
