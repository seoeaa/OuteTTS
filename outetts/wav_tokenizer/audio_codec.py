import torchaudio
import torch
from .encoder.utils import convert_audio
from .decoder.pretrained import WavTokenizer
import requests
from tqdm import tqdm
import os
import platform
from loguru import logger
import hashlib
import os

class AudioCodec:
    # WavTokenizer implementation: https://github.com/jishengpeng/WavTokenizer

    def __init__(self, device: str = None, model_path: str = None):
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.get_cache_dir()
        self.model_path = os.path.join(self.cache_dir, "wavtokenizer_large_speech_320_24k.ckpt")
        self.config_path = self.get_config_path()
        self.model_url = "https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt"
        self.expected_hash = "7450020c154f6aba033cb8651466cb79cb1b1cdd10ea64eaba68e7871cabcc5a"
        if model_path is None:
            self.ensure_model_exists()
        else:
            self.model_path = model_path
            if os.path.isdir(self.model_path):
                raise ValueError(f"Model path {self.model_path} is a directory. Please provide a valid model file path.")
            if not self.check_file_hash(self.model_path, self.expected_hash):
                raise ValueError(f"Model file {self.model_path} has an invalid hash. Please check the file integrity.")

        try:
            self.wavtokenizer = WavTokenizer.from_pretrained0802(self.config_path, self.model_path)
        except Exception as e:
            logger.error(e)
            raise ImportError(f"Failed to load WavTokenizer model.")

        self.wavtokenizer = self.wavtokenizer.to(self.device)
        self.sr = 24000
        self.bandwidth_id = torch.tensor([0])

    def get_config_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "wavtokenizer_config.yaml")

    def get_cache_dir(self):
        return os.path.join(
            os.getenv('APPDATA') if platform.system() == "Windows" else os.path.join(os.path.expanduser("~"), ".cache"),
            "outeai", "tts", "wavtokenizer_large_speech_75_token")
    
    def check_file_hash(self, file_path, expected_hash):
        hash_func = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        file_hash = hash_func.hexdigest()
        return file_hash == expected_hash

    def ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading WavTokenizer model from {self.model_url}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.download_file(self.model_path, self.model_url)

        if not self.check_file_hash(self.model_path, self.expected_hash):
            logger.warning(f"Model hash mismatch, re-downloading from {self.model_url}")
            self.download_file(self.model_path, self.model_url)

    def download_file(self, save_path: str, url: str):
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as file, tqdm(
                desc=save_path,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        except Exception as e:
            logger.error(e)
            raise ConnectionError(f"Failed to download file from {url}")

    def convert_audio_tensor(self, audio: torch.Tensor, sr):
        return convert_audio(audio, sr, self.sr, 1)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        return self.convert_audio_tensor(wav, sr).to(self.device)

    def encode(self, audio: torch.Tensor):
        _,discrete_code= self.wavtokenizer.encode_infer(audio, bandwidth_id=torch.tensor([0]).to(self.device))
        return discrete_code

    def decode(self, codes):
        features = self.wavtokenizer.codes_to_features(codes)
        audio_out = self.wavtokenizer.decode(features, bandwidth_id=torch.tensor([0]).to(self.device))
        return audio_out

    def save_audio(self, audio: torch.Tensor, path: str):
        torchaudio.save(path, audio.cpu(), sample_rate=self.sr, encoding='PCM_S', bits_per_sample=16)
