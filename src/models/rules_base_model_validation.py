import re
import yaml
from dataclasses import dataclass
from loguru import logger
import pandas as pd
from fuzzywuzzy import fuzz


logger.info("Init rules_base_model")


@dataclass
class RuleBasedClassifier:
    def __init__(self):
        """
        A class representing a rule-based spam classifier. The class contains methods for training and testing the model, as well as
        classifying new messages based on a set of pre-defined rules.

        """

        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            self.path_stop_words = config["stop_words"]
            self.path_dangerous_words = config["dangerous_words"]
            self.path_spam_words = config["spam_words"]

        self.stop_words = pd.read_csv(self.path_stop_words, sep=";")[
            "stop_words"
        ].tolist()
        self.dangerous_words = pd.read_csv(self.path_dangerous_words, sep=";")[
            "dangerous_words"
        ].tolist()
        self.spam_words = pd.read_csv(self.path_spam_words, sep=";")[
            "spam_words"
        ].tolist()

        self.not_spam_id = []

        self.rules = [
            {"name": "contains_link", "check": self._check_contains_link},
            {"name": "contains_stop_word", "check": self._check_contains_stop_word},
            {
                "name": "contains_dangerous_words",
                "check": self._check_contains_dangerous_words,
            },
            {"name": "contains_spam_words", "check": self._check_contains_spam_words},
            {"name": "contains_photo", "check": self._check_contains_photo},
            {"name": "contains_not_spam_id", "check": self._check_not_spam_id},
           # {
           #     "name": "contains_reply_to_message_id",
           #     "check": self._check_reply_to_message_id,
           # },
            {
                "name": "contains_special_characters",
                "check": self._check_special_characters,
            },
        ]

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters:
            X: The input features of shape (n_samples, n_features).
            y: The target labels of shape (n_samples,).

        Returns:
            None
        """
        mask_not_spam = y == 0
        self.not_spam_id = X[mask_not_spam]["from_id"].unique()
        return None

    def predict(self, X):
        """
        Predicts the scores for the given input using the trained model.

        Parameters:
            X (pandas DataFrame): The input data to predict scores for.

        Returns:
            numpy array: An array of predicted scores for the input data.
        """
        logger.info("Predicting...")
        pred_scores = []
        for index in range(len(X)):
            message = X.iloc[index, :]
            score = self._predict_message(message)
            pred_scores.append(score)
        return pred_scores

    def _predict_message(self, message):
        """
        Calculate the total score for a given message by applying a set of rules.

        Parameters:
            message (any): The message to be evaluated.

        Returns:
            float: The total score of the message based on the rules.
        """
        total_score = 0.0
        for rule in self.rules:
            total_score += rule["check"](message)
        return total_score

    def _check_contains_link(self, message):
        """
        Check if the given message contains a link and return a score indicating the presence of a link.

        :param message: The message to be checked.
        :type message: dict

        :return: The score indicating the presence of a link.
        :rtype: float
        """
        score = 0.0
        if (len(message["text"].split()) == 1) and (
            "https://t.me" in message["text"]
            or "t.me" in message["text"]
            or "telegra.ph/" in message["text"]
        ):
            score += 0.15
        return score

    def _check_contains_stop_word(self, message):
        """
        Checks if the message contains any stop words and calculates a score based on the number of stop words found.

        Parameters:
            message (dict): The message containing the text to be checked.

        Returns:
            float: The score representing the presence of stop words in the message.
        """
        score = 0.0
        for word in message["text"].split():
            if word.lower() in self.stop_words:
                score += 0.30
        return score

    def _check_contains_dangerous_words(self, message):
        """
        Checks if the given message contains any dangerous words and calculates a score based on the number of occurrences.

        Parameters:
            message (dict): The message to check for dangerous words.

        Returns:
            float: The score calculated based on the number of dangerous words found.
        """
        score = 0.0
        for word in message["text"].split():
            if word.lower() in self.dangerous_words:
                score += 0.15
        return score

    def _check_contains_spam_words(self, message):
        """
        Checks if the given message contains the phrase "читать продолжение" and returns a score based on the result.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The score, which is incremented by 1.0 if the phrase is found.
        """
        score = 0.0
        for words in self.spam_words:
            if fuzz.partial_ratio(words, message["text"].lower()) >= 80:
                score += 0.50
        return score

    def _check_contains_photo(self, message):
        """
        Checks if the given message contains a photo and returns a score based on the result.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The score based on whether the message contains a photo.
        """
        score = 0.0
        if message["photo"]:
            score += 0.15
        return score

    def _check_not_spam_id(self, message):
        """
        Checks if the given message is not spam based on the `from_id` field.

        Parameters:
            message (dict): The message to check.

        Returns:
            float: The spam score of the message. If the `from_id` is in the `not_spam_id` list, the score is decreased by 1.0.
        """
        score = 0.0
        if message["from_id"] in self.not_spam_id:
            score -= 1.0
        return score

    def _check_reply_to_message_id(self, message):
        """
        Checks if the given message has a reply_to_message_id and returns a score.

        Args:
            message (dict): The message to check.

        Returns:
            float: The score calculated based on the presence of reply_to_message_id.
        """
        score = 0.0
        if message["reply_to_message_id"]:
            score -= 1.0
        return score

    def _check_special_characters(self, message):
        """
        Check if the given message contains any special characters and calculate a score based on the presence of such characters.

        Parameters:
            message (str): The message to check for special characters.

        Returns:
            float: The calculated score based on the presence of special characters.
        """
        score = 0.0
        pattern = r"[α-ωΑ-Ω]"
        result = re.search(pattern, message["text"])
        if result:
            score += 0.3
        return score
