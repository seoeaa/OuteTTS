import os
import pandas as pd
import torchaudio
from tqdm import tqdm
import torch
import json
from pathlib import Path
import shutil
from datasets import load_dataset
import re

def clean_text(text):
    """Очистка и нормализация текста"""
    # Удаляем лишние пробелы и переносы строк
    text = " ".join(text.split())
    # Удаляем специальные символы, оставляем только буквы, цифры и базовую пунктуацию
    text = re.sub(r'[^а-яА-Я0-9\s\.,!?-]', '', text)
    return text.strip()

def prepare_common_voice(output_dir: str, max_samples: int = None):
    """
    Подготовка датасета Common Voice Russian для обучения
    
    Args:
        output_dir: Директория для сохранения обработанных данных
        max_samples: Максимальное количество сэмплов (None для использования всех)
    """
    print("Загрузка Common Voice Russian...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "ru",
        split="train",
        trust_remote_code=True  # Добавили этот параметр
    )
    
    # Создаем директории
    data_dir = Path(output_dir)
    wav_dir = data_dir / "wavs"
    os.makedirs(wav_dir, exist_ok=True)
    
    # Подготовка данных
    metadata = []
    
    # Ограничиваем количество сэмплов если указано
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    print("Обработка аудиофайлов...")
    for idx, item in enumerate(tqdm(samples)):
        try:
            # Очистка текста
            text = clean_text(item['sentence'])
            if not text:
                continue
                
            # Загрузка и ресемплинг аудио
            audio_path = item['path']
            waveform, sr = torchaudio.load(audio_path)
            
            # Конвертируем в моно если стерео
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресемплинг до 24kHz
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            
            # Сохраняем wav файл
            output_path = wav_dir / f"cv_ru_{idx:06d}.wav"
            torchaudio.save(output_path, waveform, 24000)
            
            # Добавляем метаданные
            duration = waveform.shape[1] / 24000
            metadata.append({
                "audio_filepath": str(output_path),
                "text": text,
                "duration": duration
            })
            
        except Exception as e:
            print(f"Ошибка при обработке файла {idx}: {str(e)}")
            continue
    
    # Сохраняем метаданные
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Создаем файл со списком путей для обучения
    with open(data_dir / "filelist.txt", 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(f"{item['audio_filepath']}|{item['text']}\n")
    
    print(f"Обработано {len(metadata)} файлов")
    print(f"Данные сохранены в {output_dir}")
    
    return data_dir

if __name__ == "__main__":
    # Подготавливаем датасет
    output_dir = "./data/russian_common_voice"
    prepare_common_voice(output_dir, max_samples=10000)  # Начнем с 10000 сэмплов для теста
