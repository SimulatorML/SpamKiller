from dataclasses import dataclass
import logging

# import re
import numpy as np

# from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RuleBasedClassifier")


@dataclass
class RuleBasedClassifier:
    """
    A class representing a rule-based spam classifier. The class contains methods for training and testing the model, as well as
    classifying new messages based on a set of pre-defined rules.

    """

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
        pred_scores = np.empty(len(X))
        for index in range(len(X)):
            pred_scores[index] = self._predict_message(X.iloc[index])
        return pred_scores

    def _predict_message(self, message):
        """
        This function takes a message and returns a score based on certain criteria. It checks for the presence of
        stop words and dangerous words in the message, as well as the length of the message and the presence of certain
        URLs. It then calculates a score based on these criteria and returns it.
        The rules are based on the analysis of the dataset train without peeping in the test.

        :param self: The object instance that the method is called on.
        :param message: A dictionary containing the message to be checked.
        :return: A float representing the score of the message based on the criteria.
        """
        stop_word = [
            "вайлдберриз",
            "вайлдбериз",
            "вб",
            "сво",
            "cvo",
            "сbо",
            "cβо",
            "путин",
            "зеленский",
            "немцы",
            "пытали",
            "пᴩигoжин",
            "взломы",
            "взлом",
            "цензуры",
            "цензура",
            "фашисты",
            "халявно",
            "халява",
        ]
        dangerous_word = [
            "продолжение",
            "подробнее",
            "даром",
            "заработок",
            "крипта",
            "крипты",
            "взлом",
            "взломы",
            "лс",
            "хакеров",
            "хакеры",
        ]
        score = []
        # if "https://t.me" in message['text'] or "t.me" in message['text'] or "https://" in message['text']: # Check if message contains link
        # score.append(0.05)
        if len(message["text"].split()) == 1 and (
            "https://t.me" in message["text"]
            or "t.me" in message["text"]
            or "https://" in message["text"]
            or "telegra.ph/" in message["text"]
        ):  # Check if message contains link
            score.append(0.15)
        # if re.search("[а-я]", message['text']) and re.search("[a-z]", message['text']): # Check if message contains mixed alphabet
        # score.append(0.05)
        for word in message["text"].split():
            if word.lower() in stop_word:  # Check stop words
                score.append(0.30)
            if word.lower() in dangerous_word:  # Check dangerous words
                score.append(0.10)
        if message["photo"]:
            score.append(0.15)
        if "читать продолжение" in message["text"]:
            score.append(1.0)
        return sum(score)
        # первое сообщение от пользователя
        # ответ на чьё-то сообщение
        # сколько сообщений пользователь отправлял до этого
        # большие и маленькие слова
        # язык кроме русского и английского
        # Check if message contains stop words with fuzzywuzzy
        # def contains_stop_word(message):
        #     if message is None:
        #         return False
        #     for spam_message in example_spam:
        #         if fuzz.ratio(message, spam_message) >= 80:
        #             return True
        #     return False
