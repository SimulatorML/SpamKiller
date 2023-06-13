import pandas as pd
from loguru import logger
import re
from fuzzywuzzy import fuzz
import yaml


class Model:
    def __init__(self):
        with open("./config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
            cleaned_spam_path = config["path_cleaned_spam"]
            clened_not_spam_path = config["path_cleaned_not_spam"]
            self.df_cleaned_spam_and_not_spam_path = config["df_cleaned_spam_and_not_spam"]

        cleaned_spam = pd.read_csv(cleaned_spam_path, sep=";")
        clened_not_spam = pd.read_csv(clened_not_spam_path, sep=";")

        self.df_cleaned_spam_and_not_spam = pd.concat(
            [cleaned_spam, clened_not_spam], ignore_index=True
        )

        example_spam = cleaned_spam["text"].astype(str).tolist()
        self.example_spam = example_spam

    def contains_stop_word(self, message):
        if message is None:
            return False
        return message in self.example_spam

    # Check if message contains stop words with fuzzywuzzy
    # def contains_stop_word(message):
    #     if message is None:
    #         return False
    #     for spam_message in example_spam:
    #         if fuzz.ratio(message, spam_message) >= 80:
    #             return True
    #     return False

    def contains_link(self, message):
        return "https://t.me" in message or "t.me" in message or "https://" in message

    def contains_mixed_alphabet(self, message):
        text = message
        if re.search("[а-я]", text) and re.search("[a-z]", text):
            return True
        return False

    def handle_msg_with_args(self, message):
        raw_scores = []
        if self.contains_stop_word(message):
            raw_scores.append(0.90)
        if self.contains_link(message):
            raw_scores.append(0.05)
        if self.contains_mixed_alphabet(message):
            raw_scores.append(0.05)

        score = round(sum(raw_scores) / 1, 2)
        return score

    def predict(self, messages):
        pred_scores = [self.handle_msg_with_args(msg) for msg in messages]
        return pred_scores
