from dataclasses import dataclass
from loguru import logger
import pandas as pd
import yaml
import re

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
            self.path_stop_word = config["stop_word"]
            self.path_dangerous_word = config["dangerous_word"]
        self.stop_word = pd.read_csv(self.path_stop_word, sep=";")['stop_word'].tolist()
        self.dangerous_word = pd.read_csv(self.path_dangerous_word, sep=";")['dangerous_word'].tolist()

        self.rules = [
            {"name": "contains_link", "check": self._check_contains_link, "score": 0.15},
            {"name": "contains_stop_word", "check": self._check_contains_stop_word, "score": 0.30},
            {"name": "contains_dangerous_word", "check": self._check_contains_dangerous_word, "score": 0.10},
            {"name": "contains_read_more", "check": self._check_contains_read_more, "score": 1.0},
            {"name": "contains_mixed_alphabet", "check": self._contains_mixed_alphabet, "score": 0.80},
            # add new rules here...
        ]

    def fit(self, X, y):
        """
        This function is present for API consistency by convention, but it is not used.

        Parameters:
        X: The input data, which is not used by this function.
        y: The target data, which is also not used by this function.

        Returns:
        None
        """
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
        for message in X:
            score = self._predict_message(message)
            pred_scores.append(score)
        return pred_scores

    def _predict_message(self, message):
        total_score = 0.0
        max_score = sum(rule["score"] for rule in self.rules)
        for rule in self.rules:
            if rule["check"](message):
                total_score += rule["score"]
        normalized_score = total_score / max_score if max_score != 0 else 0
        return round(normalized_score, 2)

    def _check_contains_link(self, message):
        if len(message.split()) == 1 and (
            "https://t.me" in message
            or "t.me" in message
            or "https://" in message
            or "telegra.ph/" in message
        ):
            return True
        return False

    def _check_contains_stop_word(self, message):
        for word in message.split():
            if word.lower() in self.stop_word:
                return True
        return False

    def _check_contains_dangerous_word(self, message):
        for word in message.split():
            if word.lower() in self.dangerous_word:
                return True
        return False

    def _check_contains_read_more(self, message):
        if "читать продолжение" in message:
            return True
        return False

    def _contains_mixed_alphabet(self, message):
        text = message
        if re.search("[а-я]", text) and re.search("[a-z]", text):
            return True
        return False

    # Add new rule check methods here...
