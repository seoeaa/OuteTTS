import re
from typing import List
import pypinyin
from transliterate import translit
import num2words

class RussianTextProcessor:
    """
    Процессор для обработки русского текста с улучшенной транслитерацией
    и фонетическими правилами
    """
    
    def __init__(self):
        self.numbers_re = re.compile(r'\d+([.,]\d+)?')
        self.cleanup_re = re.compile(r'[^а-яА-Я\s.,!?-]')
        
        # Правила для улучшенной транслитерации
        self.phonetic_rules = {
            'я': 'ya',
            'ю': 'yu',
            'е': 'ye',
            'ё': 'yo',
            'ж': 'zh',
            'ш': 'sh',
            'щ': 'sch',
            'ц': 'ts',
            'ч': 'ch',
            'й': 'y',
            'ь': '',  # Мягкий знак - особая обработка
            'ъ': '',  # Твёрдый знак - особая обработка
            'э': 'e',
            'ы': 'y'
        }
        
        # Правила для позиционных изменений
        self.positional_rules = {
            'о': {'unstressed': 'a'},  # Безударная о -> а
            'е': {'unstressed': 'i'},  # Безударная е -> и
            'я': {'unstressed': 'yi'}, # Безударная я -> йи
            'ть ': 't ',  # Мягкое окончание глаголов
            'тся': 'tsa',  # Возвратные глаголы
            'ться': 'tsa'  # Возвратные глаголы в инфинитиве
        }

    def _convert_numbers(self, text: str) -> str:
        """Конвертирует числа в текстовое представление"""
        def replace_number(match):
            number = match.group(0)
            try:
                # Заменяем точку на запятую для правильной обработки
                number = number.replace(',', '.')
                return num2words.num2words(float(number), lang='ru')
            except:
                return number
        
        return self.numbers_re.sub(replace_number, text)

    def _apply_phonetic_rules(self, text: str) -> str:
        """Применяет фонетические правила транслитерации"""
        result = ''
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i:i+2] in self.positional_rules:
                result += self.positional_rules[text[i:i+2]]
                i += 2
            elif text[i] in self.phonetic_rules:
                result += self.phonetic_rules[text[i]]
                i += 1
            else:
                result += text[i]
                i += 1
        return result

    def _handle_stress_patterns(self, text: str) -> str:
        """
        Обрабатывает ударения и редукцию гласных.
        В идеале здесь должен быть словарь ударений или ML модель.
        """
        # Упрощенная версия - считаем первую гласную ударной
        words = text.split()
        result = []
        
        for word in words:
            # Простая эвристика: первая гласная ударная
            vowels_found = 0
            processed_word = ''
            for char in word:
                if char in 'аеёиоуыэюя':
                    if vowels_found == 0:
                        # Ударная гласная - оставляем как есть
                        processed_word += char
                    else:
                        # Безударная гласная - применяем редукцию
                        if char in self.positional_rules and 'unstressed' in self.positional_rules[char]:
                            processed_word += self.positional_rules[char]['unstressed']
                        else:
                            processed_word += char
                    vowels_found += 1
                else:
                    processed_word += char
            result.append(processed_word)
        
        return ' '.join(result)

    def process(self, text: str) -> str:
        """
        Основной метод обработки текста
        """
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Конвертируем числа в текст
        text = self._convert_numbers(text)
        
        # Очищаем от лишних символов
        text = self.cleanup_re.sub('', text)
        
        # Обрабатываем ударения и редукцию
        text = self._handle_stress_patterns(text)
        
        # Применяем фонетические правила
        text = self._apply_phonetic_rules(text)
        
        # Финальная очистка
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_words(self, text: str) -> List[str]:
        """
        Обрабатывает текст и возвращает список слов
        """
        processed_text = self.process(text)
        return processed_text.split()

# Пример использования:
if __name__ == "__main__":
    processor = RussianTextProcessor()
    
    # Тестовые примеры
    test_texts = [
        "Привет, как дела?",
        "Я люблю программирование!",
        "Число 42 - ответ на главный вопрос.",
        "Съешь ещё этих мягких французских булок.",
    ]
    
    for text in test_texts:
        processed = processor.process(text)
        print(f"Исходный текст: {text}")
        print(f"Обработанный текст: {processed}")
        print("---")
