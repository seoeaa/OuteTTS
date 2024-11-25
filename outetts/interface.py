import torch
from .version.v1.interface import InterfaceHF as _InterfaceHF_v1
from .version.v1.interface import InterfaceGGUF as _InterfaceGGUF_v1
from .version.v1.interface import HFModelConfig as HFModelConfig_v1
from .version.v1.interface import GGUFModelConfig as GGUFModelConfig_v1
from loguru import logger

MODEL_CONFIGS = {
    "0.1": {
        "tokenizer": "OuteAI/OuteTTS-0.1-350M",
        "sizes": ["350M"],
        "links": ["https://huggingface.co/OuteAI/OuteTTS-0.1-350M", "https://huggingface.co/OuteAI/OuteTTS-0.1-350M-GGUF"],
        "languages": ["en"],
        "hf_interface": _InterfaceHF_v1,
        "gguf_interface": _InterfaceGGUF_v1,
    },
    "0.2": {
        "tokenizer": "OuteAI/OuteTTS-0.2-500M",
        "sizes": ["500M"],
        "links": ["https://huggingface.co/OuteAI/OuteTTS-0.2-500M", "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF"],
        "languages": ["en", "ja", "ko", "zh"],
        "hf_interface": _InterfaceHF_v1,
        "gguf_interface": _InterfaceGGUF_v1,
    },
}

def display_available_models():
    print("\n=== Available OuteTTS Models ===\n")
    separator = "-" * 50
    for version, details in MODEL_CONFIGS.items():
        print(separator)
        print(f"Version: {version}")
        print(f"Supported Languages: {', '.join(details['languages'])}")
        print(f"Model Sizes: {', '.join(details['sizes'])}")
        print(f"Available Formats: HF, GGUF")
        print(f"Tokenizer: {details['tokenizer']}")
        print(f"Links: {', '.join(details['links'])}")
        print(separator + "\n")

def get_model_config(version: str):
    """
    Retrieve the configuration for a given model version.
    """
    if version not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model version '{version}'. Supported versions are: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[version]

def InterfaceHF(
        model_version: str,
        cfg: HFModelConfig_v1
    ) -> _InterfaceHF_v1:
    """
    Creates and returns a Hugging Face model interface for OuteTTS.

    Parameters
    ----------
    model_version : str
        Version identifier for the model to be loaded
    cfg : HFModelConfig_v1
        Configuration object containing parameters

    Returns
    -------
    An instance of interface based on the specified version.
    """

    config = get_model_config(model_version)
    cfg.tokenizer_path = cfg.tokenizer_path or config["tokenizer"]
    languages = config["languages"]
    if cfg.language not in languages:
        raise ValueError(f"Language '{cfg.language}' is not supported by model version '{model_version}'. Supported languages are: {languages}")
    cfg.languages = languages

    interface_class = config["hf_interface"]

    return interface_class(cfg)

def InterfaceGGUF(
        model_version: str,
        cfg: GGUFModelConfig_v1
    ) -> _InterfaceGGUF_v1:
    """
    Creates and returns a GGUF model interface for OuteTTS.

    Parameters
    ----------
    model_version : str
        Version identifier for the model to be loaded
    cfg : GGUFModelConfig_v1
        Configuration object containing parameters

    Returns
    -------
    An instance of interface based on the specified version.
    """

    if not cfg.model_path.lower().endswith('.gguf'):
        raise ValueError(f"Model path must point to a .gguf file, got: '{cfg.model_path}'")

    config = get_model_config(model_version)
    cfg.tokenizer_path = cfg.tokenizer_path or config["tokenizer"]
    languages = config["languages"]
    if cfg.language not in languages:
        raise ValueError(f"Language '{cfg.language}' is not supported by model version '{model_version}'. Supported languages are: {languages}")
    cfg.languages = languages

    interface_class = config["gguf_interface"]
    return interface_class(cfg)
