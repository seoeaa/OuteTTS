import logging
import os
import sys
import locale

import torch
from pytorch_lightning.cli import LightningCLI

# Устанавливаем кодировку UTF-8
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

def get_gpu_info():
    if not torch.cuda.is_available():
        return 0, "GPU недоступен"
    
    count = torch.cuda.device_count()
    if count == 0:
        return 0, "GPU недоступен"
    
    device_name = torch.cuda.get_device_name(0)
    return count, device_name

def main():
    # Отключаем детерминированность CUDA операций
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Установка точности для тензорных вычислений
    torch.set_float32_matmul_precision('high')
    
    # Получение информации о GPU
    gpu_count, gpu_name = get_gpu_info()
    logger.info(f"Доступно GPU: {gpu_count}")
    if gpu_count > 0:
        logger.info(f"Используется: {gpu_name}")
    
    # Запуск обучения
    logger.info("Запуск обучения...")
    cli = LightningCLI(
        args=sys.argv[1:],
        save_config_callback=None,
    )

if __name__ == "__main__":
    main()
