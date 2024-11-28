from transformers import AutoTokenizer
from loguru import logger
import re
import inflect
import uroman as ur
import MeCab
from .russian_processor import RussianTextProcessor

class PromptProcessor:
    def __init__(self, tokenizer_path: str, languages: list[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.bos = "<|im_start|>"
        self.eos = "<|im_end|>"
        self.special_tokens = {
            "audio_code": "<|{}|>",
            "text_start": "<|text_start|>",
            "text_end": "<|text_end|>",
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "time": "<|t_{:.2f}|>",
            "code_start": "<|code_start|>",
            "code_end": "<|code_end|>",
            "text_sep": "<|text_sep|>"
        }
        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n"
        self.map_audio_tokens = self.get_audio_token_map()

        self.lec = inflect.engine()
        self.uroman = ur.Uroman()
        self.wakati = MeCab.Tagger("-Owakati")
        self.wakati_use = ["ja", "zh", "ko"]
        self.languages = languages
        
        # Инициализируем русский процессор
        self.russian_processor = RussianTextProcessor()

    def get_audio_token_map(self) -> dict:
        return {
            self.tokenizer.encode(self.special_tokens["audio_code"].format(i), add_special_tokens=False)[0]: i
            for i in range(4100)
        }

    def process_text(self, text: str, language: str):
        if language not in self.languages:
            raise ValueError(f"Language {language} not supported, supported languages are {self.languages}")
        
        logger.info(f"Processing text for language {language}. Input text: {text}")
        
        # Специальная обработка для русского языка
        if language == "ru":
            words = self.russian_processor.process_words(text)
            logger.info(f"Processed Russian text: {' '.join(words)}")
            return words
        
        # Обработка для других языков
        elif language != "en":
            if language in self.wakati_use:
                text = self.wakati.parse(text)
            text = self.uroman.romanize_string(text)
        
        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[-_/,\.\\]', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.split()

    def create_audio_prompt(self, words: list) -> str:
        prompt = []
        logger.info("Creating audio prompt. Input words:")
        for i in words:
            logger.info(f"Processing word: {i}")
            word = i["word"]
            duration = self.special_tokens["time"].format(i["duration"])
            tokens = "".join([self.special_tokens["audio_code"].format(c) for c in i["codes"]])
            prompt_line = f'{word}{duration}{self.special_tokens["code_start"]}{tokens}{self.special_tokens["code_end"]}'
            logger.info(f"Generated prompt line: {prompt_line}")
            prompt.append(prompt_line)
        final_prompt = "\n".join(prompt)
        logger.info(f"Final audio prompt: {final_prompt}")
        return final_prompt
        
    def get_completion_prompt(self, text: str, language: str, speaker: dict = None) -> str:
        logger.info(f"Getting completion prompt for text: {text}, language: {language}")
        if speaker:
            logger.info(f"Speaker info: {speaker}")
            
        words = self.process_text(text, language)
        if speaker is not None:
            if speaker["language"] != language:
                logger.warning(f"Speaker language {speaker['language']} does not match text language {language}")
            speaker_text = speaker["text"]
            logger.info(f"Processing speaker text: {speaker_text}")
            words = self.process_text(speaker_text, speaker["language"]) + words

        words = f"{self.special_tokens['text_sep']}".join([i.strip() for i in words])
        logger.info(f"Processed words joined with separator: {words}")

        prompt = self.text_prompt.format(
            bos=self.bos, 
            text_start=self.special_tokens['text_start'], 
            words=words, 
            text_end=self.special_tokens['text_end'],
            audio_start=self.special_tokens['audio_start']
        )
        logger.info(f"Base prompt created: {prompt}")

        if speaker is not None:
            logger.info("Adding speaker audio prompt")
            audio_prompt = self.create_audio_prompt(speaker["words"])
            prompt += audio_prompt
            logger.info(f"Added audio prompt. Final prompt: {prompt}")

        return prompt
    
    def get_training_prompt(self, text: str, language: str, speaker: dict) -> str:
        words = self.process_text(text, language)
        words = f"{self.special_tokens['text_sep']}".join([i.strip() for i in words])

        prompt = self.text_prompt.format(
            bos=self.bos, 
            text_start=self.special_tokens['text_start'], 
            words=words, 
            text_end=self.special_tokens['text_end'],
            audio_start=self.special_tokens['audio_start']
        )
        prompt += self.create_audio_prompt(speaker["words"])
        prompt += f"\n{self.special_tokens['audio_end']}\n{self.eos}\n"

        return prompt
    
    def extract_audio_from_tokens(self, tokens: list[int]) -> list[int]:
        return [self.map_audio_tokens[i] for i in tokens if i in self.map_audio_tokens]
