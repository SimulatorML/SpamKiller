import re
import emoji
import yaml
from dataclasses import dataclass
from loguru import logger
import pandas as pd
from fuzzywuzzy import fuzz


@dataclass
class RuleBasedClassifier:
    def __init__(self):
        """
        A class representing a rule-based spam classifier. The class contains methods for training and testing the model, as well as
        classifying new messages based on a set of pre-defined rules.

        """

        with open("./config.yml", "r") as f:
            config = yaml.safe_load(f)
            self.path_stop_words = config["stop_words"]
            self.path_dangerous_words = config["dangerous_words"]
            self.path_spam_words = config["spam_words"]
            self.path_words_fuzzy_not_enough = config["words_fuzzy_not_enough"]
            self.path_not_spam_id = config["path_not_spam_id"]
            self.path_whitelist_urls = config["whitelist_urls"]

        self.stop_words = pd.read_csv(self.path_stop_words, sep=";")[
            "stop_words"
        ].tolist()
        self.dangerous_words = pd.read_csv(self.path_dangerous_words, sep=";")[
            "dangerous_words"
        ].tolist()
        self.spam_words = pd.read_csv(self.path_spam_words, sep=";")[
            "spam_words"
        ].tolist()
        self.words_fuzzy_not_enough = pd.read_csv(
            self.path_words_fuzzy_not_enough, sep=";"
        )["words_fuzzy_not_enough"].tolist()
        self.not_spam_id = pd.read_csv(self.path_not_spam_id, sep=";")[
            "not_spam_id"
        ].tolist()
        self.whitelist_urls = pd.read_csv(self.path_whitelist_urls, sep=";")[
            "whitelist_urls"
        ].tolist()

        self.rules = [
            {
                "name": "contains_telegram_link",
                "check": self._check_contains_telegram_link,
            },
            {"name": "contains_stop_word", "check": self._check_contains_stop_word},
            {
                "name": "contains_dangerous_words",
                "check": self._check_contains_dangerous_words,
            },
            {"name": "contains_spam_words", "check": self._check_contains_spam_words},
            {
                "name": "contains_cyrillic_spoofing",
                "check": self._check_contains_cyrillic_spoofing,
            },
            {"name": "contains_photo", "check": self._check_contains_photo},
            {"name": "contains_not_spam_id", "check": self._check_not_spam_id},
            {
                "name": "contains_special_characters",
                "check": self._check_special_characters,
            },
            {"name": "check_len_message", "check": self._check_len_message},
            {
                "name": "contains_words_fuzzy_not_enough",
                "check": self._check_words_fuzzy_not_enough,
            },
            {
                "name": "contains_сapital_letters",
                "check": self._check_capital_letters,
            },
            {
                "name": "contains_emoji",
                "check": self._contains_emoji,
            },
            {
                "name": "contains_url",
                "check": self._check_contains_url,
            },
        ]

        logger.info("Initialized RuleBasedClassifier")

    def predict(self, X):
        """
        Predicts the scores for the given input using the trained model.

        Parameters:
            X (pandas DataFrame): The input data to predict scores for.

        Returns:
            numpy array: An array of predicted scores for the input data.
        """

        logger.info("Predicting...")
        total_score = 0
        total_feature = ""
        
        # Добавляем проверку форвард-сообщений
        score, feature = self._check_forward_spam(X.iloc[0])
        total_score += score
        total_feature += feature
        
        # Существующие проверки
        for check_method in [
            self._check_contains_telegram_link,
            self._check_contains_stop_word,
            self._check_contains_dangerous_words,
            self._check_contains_spam_words,
            self._check_contains_cyrillic_spoofing,
            self._contains_emoji,
            self._check_special_characters,
            self._check_not_spam_id,
        ]:
            score, feature = check_method(X.iloc[0])
            total_score += score
            total_feature += feature

        return self._normalize_score(total_score), total_feature

    def _normalize_score(self, score, threshold = 0.8):
        """
        Normalize the score to a range from 0 to 1 using a threshold value.

        Parameters:
            score (float): The input score.
            threshold (float): The threshold value above which the score is considered maximum (1).

        Returns:
            float: The normalized score.
        """
        if score >= threshold:
            normalized_score = 2 #сообщение точно спам
        elif 0.4 <= score < threshold:
            normalized_score = 1 #сообщение может быть спамом
        else:
            normalized_score = 0 #сообщение точно не спам

        return normalized_score

    def _check_contains_telegram_link(self, message):
        text = message["text"].strip()
        score = 0.0
        feature = ""
        link_pattern = re.compile(
            r"https?:\/\/(?:t\.me|telegra\.ph)\/[^\s]+|"  # обычные http и https ссылки
            r"@[\w\d_]+|"  # @username формат
            r"t\.me/\S+|"  # t.me ссылки
            r"t\.me/joinchat/\S+|"  # t.me ссылки на группы
            r"telegra\.ph/\S+"  # telegraph ссылки
        )

        links = link_pattern.findall(text)

        if not links:
            return score, feature

        if text.strip() == links[0]:
            score += 0.3
            feature = "[+0.3] - В сообщении содержится только telegram ссылка\n"
        elif len(links) >= 2:
            score += 0.3
            feature = "[+0.3] - В сообщении содержится много telegram ссылок\n"
        else:
            score += 0.15
            feature = "[+0.15] - В сообщении содержится telegram ссылка\n"

        return score, feature

    def _check_contains_stop_word(self, message):
        """
        Checks if the message contains any stop words and calculates a score based on the number of stop words found.

        Parameters:
            message (dict): The message containing the text to be checked.

        Returns:
            float: The score representing the presence of stop words in the message.
        """
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        for words in self.stop_words:
            if fuzz.token_set_ratio(words.lower(), text.lower()) >= 77:
                score += 0.30
                feature += f'[+0.3] - В сообщении содержится: "{words}"\n'

        return score, feature

    def _check_contains_dangerous_words(self, message):
        """
        Checks if the given message contains any dangerous words and calculates a score based on the number of occurrences.

        Parameters:
            message (dict): The message to check for dangerous words.

        Returns:
            float: The score calculated based on the number of dangerous words found.
        """
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        for words in self.dangerous_words:
            if fuzz.token_set_ratio(words.lower(), text.lower()) >= 77:
                score += 0.15
                feature += f'[+0.15] - В сообщении содержится: "{words}"\n'

        return score, feature

    def _check_contains_spam_words(self, message):
        """
        Checks if the given message contains the phrase "читать продолжение" and returns a score based on the result.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The score, which is incremented by 1.0 if the phrase is found.
        """
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        for words in self.spam_words:
            if fuzz.token_set_ratio(words.lower(), text.lower()) >= 90:
                score += 0.5
                feature += f'[+0.5] - В сообщении содержится: "{words}"\n'

        return score, feature

    def _check_contains_cyrillic_spoofing(self, message):
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""

        # Объединяем паттерны в один с именованными группами
        pattern = re.compile(
            r'(?P<cyr_lat_cyr>[а-яА-Я]+[a-zA-Z]+[а-яА-Я]+)|'
            r'(?P<lat_cyr_lat>[a-zA-Z]+[а-яА-Я]+[a-zA-Z]+)'
        )

        spoofed_words = set()  # Используем set для уникальных значений
        for word in text.split():
            if pattern.search(word):
                spoofed_words.add(word)
            
        if len(spoofed_words) > 0:
            score = min(0.1 * len(spoofed_words), 0.3)  # Ограничиваем максимальный score
            feature = f'[+{round(score, 1)}] - Подмена кириллицы ({", ".join(list(spoofed_words)[:3])})\n'

        return score, feature
    
    def _contains_emoji(self, message):
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""

        emojis = [char for char in text if char in emoji.EMOJI_DATA]

        if emojis:
            score += 0.15 * len(emojis)
            feature += f"[+{round(score, 2)}] - Спам эмодзи ({', '.join(emojis[:3])})\n"
        return score, feature

    def _check_contains_photo(self, message):
        """
        Checks if the given message contains a photo and returns a score based on the result.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The score based on whether the message contains a photo.
        """
        score = 0.0
        feature = ""
        if message["photo"]:
            score += 0.15
            feature = "[+0.15] - В сообщении содержится фотография\n"

        return score, feature

    def _check_not_spam_id(self, message):
        """
        Checks if the given message is not spam based on the `from_id` field.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The spam score of the message. If the `from_id` is in the `not_spam_id` list, the score is decreased by 1.0.
        """
        score = 0.0
        feature = ""
        if message["from_id"] in self.not_spam_id:
            score -= 0.5
            feature = "[-0.5] - Пользователь ранее не писал спам\n"

        return score, feature

    def _check_special_characters(self, message):
        """
        Check if the given message contains any special characters and calculate a score based on the presence of such characters.

        Parameters:
            message (str): The message to check for special characters.

        Returns:
            float: The calculated score based on the presence of special characters.
        """
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        pattern = r"[^a-zA-Zа-яА-ЯёЁ0-9.,…!?;:()[\]{}@+=*\/%<>^«»&|`\-⁃–—'\"”“‘’©#№$€_~ \t\n\r∞≈≤≥±∓√∛∜∫∑∏∂∇×÷⇒⇐⇔\\^_&∧∨¬⊕⊖⊗⊘∈∉∪∩⊆⊇⊂⊃ℕℤℚℝℂ→↦^\U00000000-\U0010ffff]"
        result = re.findall(pattern, text.lower())
        if result:
            score += len(result) * 0.1
            feature = f'[+{round(len(result) * 0.1, 1)}] - Неразрешенные символы ({", ".join(result[:3])})\n'

        return score, feature

    def _check_len_message(self, message):
        """
        Calculate the score for the length of the message.

        Parameters:
            message (dict): A dictionary containing the message text.

        Returns:
            float: The score for the length of the message.
        """
        score = 0.0
        feature = ""
        if len(message["text"]) < 5 and len(message["text"]) != 0:
            score += 0.1
            feature = "[+0.1] - Сообщение чересчур короткое\n"

        return score, feature

    def _check_words_fuzzy_not_enough(self, message):
        """
        Calculate the score for a given message based on the presence of words in the 'words_fuzzy_not_enough' list.

        Parameters:
            message (dict): A dictionary containing the message text.

        Returns:
            float: The calculated score based on the presence of words from 'words_fuzzy_not_enough' list in the message text.
        """
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        for word_fuzzy_not_enough in self.words_fuzzy_not_enough:
            for word in text.split():
                if word_fuzzy_not_enough == re.sub(
                    r"[^a-zа-я]", "", word.lower().strip()
                ):
                    score += 0.15

        return score, feature

    def _check_capital_letters(self, message):
        """
        Calculates the score based on the presence of capital letters in the input message.

        Parameters:
            message (dict): The input message containing the text.

        Returns:
            float: The calculated score.

        """
        text = message["text"].strip()
        score = 0.0
        feature = ""
        capital_pattern = "[A-ZА-Я]"
        pattern = "[a-zA-Zа-яА-Я]"

        capital_letters = re.findall(capital_pattern, text)
        letters = re.findall(pattern, text)
        try:
            if len(capital_letters) / len(letters) > 0.4 and len(text) > 5:
                score += 0.15
                feature = "[+0.15] - Большая концентрация заглавных букв\n"
        except ZeroDivisionError:
            pass

        return score, feature

    def _contains_emoji(self, message):
        text = (message["text"] + "    " + message.get("bio", "")).strip()
        score = 0.0
        feature = ""
        # Unicode кодовые точки для эмодзи
        emojis_code = [
            "\U00002757",  # Red Exclamation Mark
            "\U00002753",  # Question Mark Ornament
            "\U0001F4A6",  # Sweat Droplets
            "\U0001F4B5",  # Dollar Banknote
            "\U0001F4B0",  # Money Bag
            "\U0001F4A7",  # Droplet
            "\U0001F346",  # Eggplant
            "\U0001F34C",  # Banana
            "\U0001F351",  # Peach
            "\U0001F353",  # Strawberry
            "\U0001F352",  # Cherries
            "\U0001F608",  # Smiling Face with Horns
            "\U00002705",  # White Heavy Check Mark
            "\U0001F381",  # Wrapped Present
            "\U0001F48E",  # Gem Stone
            "\U0001F911",  # Money-Mouth Face
            "\U00002728",  # Sparkles
            "\U0001F6A8",  # Police Cars Revolving Light
        ]

        emoji_pattern = re.compile("|".join(emojis_code))
        found_emojis = emoji_pattern.findall(text)

        if found_emojis:
            score += 0.15 * len(found_emojis)
            feature += f"[+{round(score, 2)}] - Содержатся подозрительные эмодзи ({', '.join(found_emojis[:3])})\n"

        emojis = [char for char in text if char in emoji.EMOJI_DATA]

        if len(emojis) > 10:
            score += 0.15 * len(emojis)
            feature += f"[+{round(score, 2)}] - Спам эмодзи ({', '.join(emojis[:3])})\n"
        return score, feature

    def _check_contains_url(self, message):
        """
        Calculates the score for a given message based on the presence
        of various types of URLs in text except whitelisted in whitelist_urls.csv

        Parameters:
            message (dict): The input message containing the text.

        Returns:
            float: The calculated score.

        """
        text = message["text"].strip()
        score = 0.0
        feature = ""
        # Regular expression pattern for finding various types of URLs in text
        url_regex = re.compile(
            r"(?:(?:http|ftp)s?://)?"  # Scheme (optional)
            r"(?:"  # Start of group for domain/IP
            r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # Domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # IPv4 address
            r")"  # End of group for domain/IP
            r"(?::\d+)?"  # Optional port
            r"(?:/?|[/?]\S*)",  # Path (optional)
            re.IGNORECASE,
        )

        raw_links = url_regex.findall(text)
        raw_links = [link.rstrip("/") for link in raw_links]
        unwanted_links = list(set(raw_links) - set(self.whitelist_urls))

        if not unwanted_links:
            return score, feature

        if text.strip() == unwanted_links[0]:
            score += 0.3
            feature = "[+0.3] - В сообщении содержится только ссылка\n"

        elif len(unwanted_links) == 1:
            score += 0.15
            feature = "[+0.15] - В сообщении содержится ссылка и текст\n"

        else:
            score += 0.3
            feature = "[+0.3] - В сообщении содержится несколько ссылок\n"

        return score, feature

    def _check_forward_spam(self, message):
        """
        Проверяет форвард-сообщения и истории на признаки спама.
        """
        score = 0.0
        feature = ""
        
        # Если это форвард
        if message.get("is_forwarded"):
            # Базовый скор за форвард
            score += 0.1
            feature += "[+0.1] - Сообщение переслано\n"
            
            # Если это история
            if message.get("is_story"):
                score += 0.2
                feature += "[+0.2] - Переслана история\n"
            
            # Проверяем ID источника форварда
            forward_from_id = message.get("forward_from_id")
            forward_from_chat_id = message.get("forward_from_chat_id")
            
            # Если источник форварда неизвестен или скрыт
            if not forward_from_id and not forward_from_chat_id:
                score += 0.2
                feature += "[+0.2] - Источник форварда скрыт\n"
            
            # Проверяем текст на спам-признаки более строго для форвард-сообщений
            text = message.get("text", "").lower()
            if any(word in text for word in self.spam_words):
                # Повышенный скор для историй
                if message.get("is_story"):
                    score += 0.4
                    feature += "[+0.4] - История содержит спам-слова\n"
                else:
                    score += 0.3
                    feature += "[+0.3] - Форвард содержит спам-слова\n"
            
        return score, feature

    def _check_story_content(self, message):
        """
        Проверяет содержимое истории на признаки спама
        """
        score = 0.0
        feature = ""
        
        if message.get("is_story"):
            # Проверяем caption истории
            story_caption = message.get("story_caption", "").lower()
            if story_caption:
                # Проверка на спам-слова
                for word in self.spam_words:
                    if word.lower() in story_caption:
                        score += 0.3
                        feature += f'[+0.3] - В описании истории найдено спам-слово "{word}"\n'
                
                # Проверка на ссылки
                link_pattern = re.compile(
                    r"https?:\/\/(?:t\.me|telegra\.ph)\/[^\s]+|"  # обычные http и https ссылки
                    r"@[\w\d_]+|"  # @username формат
                    r"t\.me/\S+|"  # t.me ссылки
                    r"t\.me/joinchat/\S+|"  # t.me ссылки на группы
                    r"telegra\.ph/\S+"  # telegraph ссылки
                )
                
                links = link_pattern.findall(story_caption)
                if links:
                    if story_caption.strip() == links[0]:
                        score += 0.4
                        feature += "[+0.4] - Описание истории содержит только ссылку\n"
                    elif len(links) >= 2:
                        score += 0.4
                        feature += "[+0.4] - Описание истории содержит несколько ссылок\n"
                    else:
                        score += 0.2
                        feature += "[+0.2] - Описание истории содержит ссылку\n"
        
            # Проверка на тип медиа
            media_type = message.get("media_type")
            if media_type == "photo":
                score += 0.1
                feature += "[+0.1] - История содержит фото\n"
            elif media_type == "video":
                score += 0.1
                feature += "[+0.1] - История содержит видео\n"
    
        return score, feature

    def check_message(self, message):
        """
        Основной метод проверки сообщения
        """
        total_score = 0.0
        total_feature = ""
        
        # Добавляем проверку историй к существующим проверкам
        story_score, story_feature = self._check_story_content(message)
        total_score += story_score
        total_feature += story_feature
        
        # Существующие проверки
        for check_method in [
            self._check_contains_telegram_link,
            self._check_contains_stop_word,
            self._check_contains_dangerous_words,
            self._check_contains_spam_words,
            self._check_contains_cyrillic_spoofing,
            self._contains_emoji,
            self._check_special_characters,
            self._check_not_spam_id,
        ]:
            score, feature = check_method(message)
            total_score += score
            total_feature += feature

        return total_score, total_feature
