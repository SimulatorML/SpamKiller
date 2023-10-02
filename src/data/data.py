import json
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import yaml
from dataclasses import dataclass
from loguru import logger
from fuzzywuzzy import fuzz
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import gc


class Data:
    @staticmethod
    def json_to_csv(json_dir: str, csv_dir: str, label: int) -> None:
        """
        Converts a JSON file to a CSV file.

        Parameters:
            json_dir (str): The directory path of the JSON file.
            csv_dir (str): The directory path to save the CSV file.
            label (int): The label to assign to each message in the CSV file.

        Returns:
            None
        """
        with open(json_dir, encoding="utf-8", newline="") as f:
            data = json.load(f)

        messages_to_write = []
        for message in data["messages"]:
            text = message.get("text")
            from_id = message.get("from_id")
            photo = "photo" in message
            reply_to_message_id = "reply_to_message_id" in message
            if isinstance(text, list):
                texts = []
                for entity in text:
                    if isinstance(entity, str):
                        texts.append(entity)
                    elif isinstance(entity, dict):
                        if "text" in entity and isinstance(entity["text"], str):
                            texts.append(entity["text"])
                text = " ".join(texts)
            elif not isinstance(text, str):
                text = ""
            messages_to_write.append([text, photo, from_id, reply_to_message_id, label])

        with open(csv_dir, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(
                ["text", "photo", "from_id", "reply_to_message_id", "label"]
            )
            writer.writerows(messages_to_write)

        print(
            f"Ready! The text from the JSON file was overwritten into a file {csv_dir}"
        )

    @staticmethod
    def clean_dataframe(file_dir: str, save_dir: str) -> None:
        """
        Clean the given dataframe by removing null values, duplicate text, and empty text entries.

        Parameters:
            file_dir (str): The directory of the CSV file to read.
            save_dir (str): The directory to save the cleaned dataframe as a CSV file.

        Returns:
            None
        """
        dataframe = pd.read_csv(file_dir, sep=";")

        dataframe = dataframe.dropna()
        dataframe["text"] = (
            dataframe["text"].str.replace("\s+", " ", regex=True).str.strip()
        )
        dataframe = dataframe.drop_duplicates(subset="text")
        dataframe = dataframe[dataframe["text"].str.len() > 0]
        # fe = FeatureEngineering(dataframe)
        #   dataframe = fe.feature_extract().drop(columns=['text', 'from_id', 'photo', 'reply_to_message_id'])
        dataframe.to_csv(save_dir, index=False, sep=";")

    @staticmethod
    def train_test_split(files_dir: list, train_dir: str, test_dir: str) -> None:
        """
        Split the given list of file directories into train and test datasets.

        Parameters:
            - files_dir (list): A list of file directories.
            - train_dir (str): The directory to save the train dataset.
            - test_dir (str): The directory to save the test dataset.

        Returns:
            None
        """
        train = []
        test = []

        for path in files_dir:
            df = pd.read_csv(path, sep=";")
            train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            train.append(train_df)
            test.append(test_df)

        train = pd.concat(train, ignore_index=True, axis=0)
        test = pd.concat(test, ignore_index=True, axis=0)

        train = train.dropna().drop_duplicates(subset="text")
        test = test.dropna().drop_duplicates(subset="text")

        train.to_csv(train_dir, sep=";", index=False)
        test.to_csv(test_dir, sep=";", index=False)


@dataclass
class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            self.path_stop_words = config["stop_words"]
            self.path_dangerous_words = config["dangerous_words"]
            self.path_spam_words = config["spam_words"]
            self.path_words_fuzzy_not_enough = config["words_fuzzy_not_enough"]

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
        self.not_spam_id = []

        self.text_features = [
            {"name": "contains_link", "check": self._check_contains_link},
            {"name": "contains_stop_word", "check": self._check_contains_stop_word},
            {
                "name": "contains_dangerous_words",
                "check": self._check_contains_dangerous_words,
            },
            {"name": "contains_spam_words", "check": self._check_contains_spam_words},
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
                "name": "contains_special_characters",
                "check": self._check_special_characters,
            },
            {
                "name": "contains_words_fuzzy_not_enough",
                "check": self._check_words_fuzzy_not_enough,
            },
            {
                "name": "num_emojis",
                "check": self._check_num_emojis,
            },
            {
                "name": "has_tg_link",
                "check": self._check_has_tg_link,
            },
            {
                "name": "num_unique_symbols",
                "check": self._check_num_unique_symbols,
            },
            {
                "name": "num_special_characters",
                "check": self._check_num_special_characters,
            },
            {
                "name": "num_capitalized_words",
                "check": self._check_num_capitalized_words,
            },
            {
                "name": "num_exclamation_marks",
                "check": self._check_num_exclamation_marks,
            },
            {
                "name": "num_question_marks",
                "check": self._check_num_question_marks,
            },
            {
                "name": "num_uppercase_letters",
                "check": self._check_num_uppercase_letters,
            },
            {
                "name": "num_non_ascii_characters",
                "check": self._check_num_non_ascii_characters,
            },
            {
                "name": "num_all_caps_words",
                "check": self._check_num_all_caps_words,
            },
            {
                "name": "num_unique_words",
                "check": self._check_num_unique_words,
            },
            {
                "name": "average_sentence_length",
                "check": self._check_average_sentence_length,
            },
            {
                "name": "num_emoticons",
                "check": self._check_num_emoticons,
            },
        ]
        self.photo_features = [
            {"name": "contains_photo", "check": self._check_contains_photo}
        ]
        self.id_features = [
            {"name": "contains_not_spam_id", "check": self._check_not_spam_id}
        ]

    def feature_extract(self):
        df = self.df.copy()
        for feature in tqdm(self.text_features):
            df[feature["name"]] = feature["check"](df["text"])
        for feature in tqdm(self.photo_features):
            df[feature["name"]] = feature["check"](df["photo"])
        for feature in tqdm(self.id_features):
            df[feature["name"]] = feature["check"](df["from_id"])
        tfidf_features = self.get_tfidf_features(df["text"])
        assert df.index.equals(tfidf_features.index), "Indexes do not match!"
        df = pd.concat([df, tfidf_features], axis=1)
        gc.collect()
        return df

    def _check_contains_link(self, df: pd.Series) -> pd.Series:
        def regular_process(message_text: str):
            # Regular expression pattern to match URLs
            count_links = 0
            url_pattern = r"(?i)\b((?:http[s]?://|www\d{0,3}[.]|telegram[.]me/|t[.]me/|telegra[.]ph/)[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))"

            # Search for URLs in the message text
            urls = re.findall(url_pattern, message_text)

            # Check if any found urls are internal Telegram links
            internal_links = [
                url for url in urls if "t.me" in url[0] or "telegra.ph/" in url[0]
            ]

            if internal_links or "none" in message_text.lower().split()[-1]:
                count_links += 1
            return count_links

        return df.apply(regular_process)

    def _check_contains_stop_word(self, df: pd.Series):
        def regular_process(message_text: str):
            count_stop_words = 0
            for words in self.stop_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_stop_words += 1
            return count_stop_words

        return df.apply(regular_process)

    def _check_contains_dangerous_words(self, df: pd.Series):
        def regular_process(message_text: str):
            count_dangerous_words = 0
            for words in self.dangerous_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_dangerous_words += 1
            return count_dangerous_words

        return df.apply(regular_process)

    def _check_contains_spam_words(self, df: pd.Series):
        def regular_process(message_text: str):
            count_spam_words = 0
            for words in self.spam_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_spam_words += 1
            return count_spam_words

        return df.apply(regular_process)

    def _check_contains_photo(self, df: pd.Series):
        return df.apply(lambda photo: 1 if photo else 0)

    def _check_not_spam_id(self, df: pd.Series):
        return df.apply(lambda from_id: -1 if int(re.sub(r'(channel|user)', '', from_id)) in self.not_spam_id else 0)

    def _check_special_characters(self, df: pd.Series):
        def regular_process(message_text: str):
            count_special_characters = 0
            pattern = "[à-üÀ-Üα-ωΑ-ΩҐЄЇІґєїі&&[^ё̰]]"
            pattern += "|[Α-Ωα-ω̰]"
            result = re.findall(pattern, message_text.lower())
            if result:
                count_special_characters = len(result)
            return count_special_characters

        return df.apply(regular_process)

    def _check_len_message(self, df: pd.Series):
        return df.apply(lambda message_text: len(message_text))

    def _check_words_fuzzy_not_enough(self, df: pd.Series):
        def regular_process(message_text: str):
            count_words_fuzzy = 0
            for word_fuzzy_not_enough in self.words_fuzzy_not_enough:
                for word in message_text.split():
                    if word_fuzzy_not_enough == re.sub(r"[^a-zа-я]", "", word.lower()):
                        count_words_fuzzy += 1
            return count_words_fuzzy

        return df.apply(regular_process)

    def _check_capital_letters(self, df: pd.Series):
        def regular_process(message_text: str):
            capital_pattern = "[A-ZА-Я]"
            capital_letters = re.findall(capital_pattern, message_text)
            return len(capital_letters)

        return df.apply(regular_process)

    def _check_num_emojis(self, df: pd.Series):
        def count_emojis(text):
            if isinstance(text, str):
                emoji_pattern = re.compile(
                    "["
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F680-\U0001F6FF"  # transport & map symbols
                    "\U0001F700-\U0001F77F"  # alchemical symbols
                    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    "\U0001FA00-\U0001FA6F"  # Chess Symbols
                    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    "\U00002702-\U000027B0"  # Dingbats
                    "\U000024C2-\U0001F251"
                    "]+",
                    flags=re.UNICODE,
                )
                emojis = emoji_pattern.findall(text)
                return len(emojis)
            else:
                return 0

        return df.apply(count_emojis)

    def _check_has_tg_link(self, df: pd.Series):
        def has_tg_link(text):
            link_pattern = re.compile(r"t\.me/[a-zA-Z0-9\-_]+")
            return int(bool(link_pattern.search(text)))

        return df.apply(has_tg_link)

    def _check_num_unique_symbols(self, df: pd.Series):
        def count_unique_symbols(text):
            unique_symbols = set(text)
            return len(unique_symbols)

        return df.apply(count_unique_symbols)

    def _check_num_special_characters(self, df: pd.Series):
        def count_special_characters(text):
            special_chars = re.findall(r"[^\w\s]", text, re.UNICODE)
            return len(special_chars)

        return df.apply(count_special_characters)

    def _check_num_capitalized_words(self, df: pd.Series):
        def count_capitalized_words(text):
            words = text.split()
            capitalized_words = [word for word in words if word[0].isupper()]
            return len(capitalized_words)

        return df.apply(count_capitalized_words)

    def _check_num_exclamation_marks(self, df: pd.Series):
        def count_exclamation_marks(text):
            return text.count("!")

        return df.apply(count_exclamation_marks)

    def _check_num_question_marks(self, df: pd.Series):
        def count_question_marks(text):
            return text.count("?")

        return df.apply(count_question_marks)

    def _check_num_uppercase_letters(self, df: pd.Series):
        def count_uppercase_letters(text):
            return sum(1 for char in text if char.isupper())

        return df.apply(count_uppercase_letters)

    def _check_num_non_ascii_characters(self, df: pd.Series):
        def count_non_ascii_characters(text):
            return sum(
                1 for char in text if ord(char) > 127 and not 1024 <= ord(char) <= 1279
            )

        return df.apply(count_non_ascii_characters)

    def _check_num_all_caps_words(self, df: pd.Series):
        def count_all_caps_words(text):
            words = text.split()
            all_caps_words = [word for word in words if word.isupper()]
            return len(all_caps_words)

        return df.apply(count_all_caps_words)

    def _check_num_unique_words(self, df: pd.Series):
        def count_unique_words(text):
            cleaned_text = re.sub(
                r"\W+", " ", text.lower()
            )  # Remove non-alphanumeric characters
            words = cleaned_text.split()
            unique_words = set(words)
            return len(unique_words)

        return df.apply(count_unique_words)

    def _check_average_sentence_length(self, df: pd.Series):
        def average_sentence_length(text):
            # Use regular expression to split text into sentences
            sentences = re.split(r"[.!?]", text)
            sentences = [
                sentence.strip() for sentence in sentences if sentence.strip()
            ]  # Remove empty sentences

            total_words = 0
            for sentence in sentences:
                total_words += len(sentence.split())

            # Calculate average sentence length
            if len(sentences) > 0:
                average_length = total_words / len(sentences)
            else:
                average_length = 0

            return average_length

        return df.apply(average_sentence_length)

    def _check_num_emoticons(self, df: pd.Series):
        def count_emoticons(text):
            emoticon_pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
            emoticons = emoticon_pattern.findall(text)
            return len(emoticons)

        return df.apply(count_emoticons)
