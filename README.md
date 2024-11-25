# OuteTTS

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.2_500M-blue)](https://huggingface.co/OuteAI/OuteTTS-0.2-500M)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.2_500M_GGUF-blue)](https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo_Space-pink)](https://huggingface.co/spaces/OuteAI/OuteTTS-0.2-500M-Demo)
[![PyPI](https://img.shields.io/badge/PyPI-OuteTTS-orange)](https://pypi.org/project/outetts/)

OuteTTS is an experimental text-to-speech model that uses a pure language modeling approach to generate speech, without architectural changes to the foundation model itself.

## Installation

```bash
pip install outetts
```

**Important:** For GGUF support, you must manually install `llama-cpp-python` first.

Visit https://github.com/abetlen/llama-cpp-python for specific installation instructions

## Usage

### Interface Usage
```python
import outetts

# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Optional: Create a speaker profile (use a 10-15 second audio clip)
# speaker = interface.create_speaker(
#     audio_path="path/to/audio/file",
#     transcript="Transcription of the audio file."
# )

# Optional: Save and load speaker profiles
# interface.save_speaker(speaker, "speaker.json")
# speaker = interface.load_speaker("speaker.json")

# Optional: Load speaker from default presets
interface.print_default_speakers()
speaker = interface.load_default_speaker(name="male_1")

output = interface.generate(
    text="Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and it can be implemented in software or hardware products.",
    # Lower temperature values may result in a more stable tone,
    # while higher values can introduce varied and expressive speech
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)

# Save the synthesized speech to a file
output.save("output.wav")

# Optional: Play the synthesized speech
# output.play()
```

### Using GGUF Model
```python
# Configure the GGUF model
model_config = outetts.GGUFModelConfig_v1(
    model_path="local/path/to/model.gguf",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
    n_gpu_layers=0,
)

# Initialize the GGUF interface
interface = outetts.InterfaceGGUF(model_version="0.2", cfg=model_config)
```

### Configure the model with bfloat16 and flash attention

```python
import outetts
import torch

model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
    dtype=torch.bfloat16,
    additional_model_config={
        'attn_implementation': "flash_attention_2"
    }
)
```

### Creating a Speaker for Voice Cloning

To achieve the best results when creating a speaker profile, consider the following recommendations:

1. **Audio Clip Duration:**
   - Use an audio clip of around **10-15 seconds**.
   - This duration provides sufficient data for the model to learn the speaker's characteristics while keeping the input manageable. The model's context length is 4096 tokens, allowing it to generate around 54 seconds of audio in total. However, when a speaker profile is included, this capacity is reduced proportionally to the length of the speaker's audio clip.

2. **Audio Quality:**
   - Ensure the audio is **clear and noise-free**. Background noise or distortions can reduce the model's ability to extract accurate voice features.

3. **Accurate Transcription:**
   - Provide a highly **accurate transcription** of the audio clip. Mismatches between the audio and transcription can lead to suboptimal results.

4. **Speaker Familiarity:**
   - The model performs best with voices that are similar to those seen during training. Using a voice that is **significantly different from typical training samples** (e.g., unique accents, rare vocal characteristics) might result in inaccurate replication.
   - In such cases, you may need to **fine-tune the model** specifically on your target speaker's voice to achieve a better representation.

5. **Parameter Adjustments:**
   - Adjust parameters like `temperature` in the `generate` function to refine the expressive quality and consistency of the synthesized voice.

## Blogs
https://www.outeai.com/blog/outetts-0.2-500m

https://www.outeai.com/blog/outetts-0.1-350m

## Credits

- WavTokenizer: [GitHub Repository](https://github.com/jishengpeng/WavTokenizer)
    - Decoder and encoder folder files are from this repository
- CTC Forced Alignment: [PyTorch Tutorial](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- Uroman: [GitHub Repository](https://github.com/isi-nlp/uroman)
    - "This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020)".
- mecab-python3 [GitHub Repository](https://github.com/SamuraiT/mecab-python3)
