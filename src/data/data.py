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
        dataframe = dataframe[dataframe["text"].str.len() > 0].dropna().drop_duplicates(subset="text")
        fe = FeatureEngineering(dataframe)
        new_dataframe = fe.feature_extract().drop(columns=['text', 'from_id', 'photo', 'reply_to_message_id'])
        new_dataframe.to_csv(save_dir, index=False, sep=";")

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
            train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['label'])

            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            train.append(train_df)
            test.append(test_df)

        train = pd.concat(train, ignore_index=True, axis=0)
        test = pd.concat(test, ignore_index=True, axis=0)

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
            df[feature['name']] = feature['check'](df['text'])
        for feature in tqdm(self.photo_features):
            df[feature['name']] = feature['check'](df['photo'])
        for feature in tqdm(self.id_features):
            df[feature['name']] = feature['check'](df['from_id'])
        return df

    def _check_contains_link(self, df: pd.Series) -> pd.Series:
        def regular_process(message_text: str):
            # Regular expression pattern to match URLs
            count_links = 0
            url_pattern = r"(?i)\b((?:http[s]?://|www\d{0,3}[.]|telegram[.]me/|t[.]me/|telegra[.]ph/)[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))"

            # Search for URLs in the message text
            urls = re.findall(url_pattern, message_text)
            
            # Check if any found urls are internal Telegram links
            internal_links = [url for url in urls if 't.me' in url[0] or 'telegra.ph/' in url[0]]
            
            if internal_links or 'none' in message_text.lower().split()[-1]:
                count_links += 0.15
            return count_links

        return df.apply(regular_process)


    def _check_contains_stop_word(self, df: pd.Series):
        def regular_process(message_text: str):
            count_stop_words = 0
            for words in self.stop_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_stop_words += 0.3
            return count_stop_words

        return df.apply(regular_process)
    
    def _check_contains_dangerous_words(self, df: pd.Series):
        def regular_process(message_text: str):
            count_dangerous_words = 0
            for words in self.dangerous_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_dangerous_words += 0.15
            return count_dangerous_words

        return df.apply(regular_process)
    
    def _check_contains_spam_words(self, df: pd.Series):
        def regular_process(message_text: str):
            count_spam_words = 0
            for words in self.spam_words:
                if fuzz.token_set_ratio(words.lower(), message_text.lower()) >= 77:
                    count_spam_words += 0.5
            return count_spam_words
        return df.apply(regular_process)

    def _check_contains_photo(self, df: pd.Series):
        return df.apply(lambda photo: 0.15 if photo else 0)

    def _check_not_spam_id(self, df: pd.Series):
        return df.apply(lambda from_id: -0.5 if from_id in self.not_spam_id else 0)

    def _check_special_characters(self, df: pd.Series):
        def regular_process(message_text: str):
            count_special_characters = 0
            pattern = "[à-üÀ-Üα-ωΑ-ΩҐЄЇІґєїі&&[^ё̰]]"
            pattern += "|[Α-Ωα-ω̰]"
            result = re.findall(pattern, message_text.lower())
            if result:
                count_special_characters = len(result) * 0.1
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
                        count_words_fuzzy += 0.15
            return count_words_fuzzy
        
        return df.apply(regular_process)

    def _check_capital_letters(self, df: pd.Series):
        def regular_process(message_text: str):
            capital_pattern = "[A-ZА-Я]"
            pattern = "[a-zA-Zа-яА-Я]"

            capital_letters = re.findall(capital_pattern, message_text)
            letters = re.findall(pattern, message_text)
            try:
                return 0.15 if len(capital_letters) / len(letters) > 0.4 and len(message_text) > 5 else 0
            except ZeroDivisionError:
                pass
            return 0
        return df.apply(regular_process)
