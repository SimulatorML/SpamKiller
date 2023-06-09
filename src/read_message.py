import pandas as pd
from loguru import logger
from aiogram import types
from src.add_new_user_id import check_user_id
import re
from fuzzywuzzy import fuzz
import yaml


# Reading the path value from the config.yml file
with open("./config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    file_path = config["cleaned_spam"]

df = pd.read_csv(file_path, sep=";")
example_spam = df["text"].astype(str).tolist()  # Converting all elements to strings


def contains_stop_word(message):
    logger.info(
        f"Checking for the presence of a stop word in a message: {message.text}"
    )
    if message.text is None:  # Checking for the existence of message.text
        return False
    for word in example_spam:  # example_spam - list of stop words
        if (
            fuzz.ratio(message.text, word) >= 80
        ):  # fuzz.ratio() returns the similarity of rows as a percentage
            return True
    return False


# Checking for the presence of a photo in a message
def contains_image(message):  # message - message object
    logger.info(f"Checking for the presence of a photo in a message")
    return message.photo is not None  # message.photo


# Checking for the presence of a link in a message
def contains_link(message):  # message - message object
    logger.info(f"Checking for a link in a message")
    return (
        "https://t.me" in message or "t.me" in message or "https://" in message
    )  # message - message object


# Checking for the presence of Cyrillic and Latin letters in one word
def contains_mixed_alphabet(message):  # message - object of the message
    logger.info(f"Checking for the presence of Cyrillic and Latin letters in one word")
    text = message.text.lower()  # message.text - message text
    if re.search("[а-я]", text) and re.search(
        "[a-z]", text
    ):  # re.search() - regular expression search to find Cyrillic and Latin letters in a word
        return True
    return False


# Message processing
logger.info(f"Message processing")


async def handle_msg_with_args(message: types.Message, bot, ADMIN_ID):
    if await check_user_id(
        message
    ):  # Checking for the presence of a user in the database
        logger.info(f"Message got from new user. Checking for spam")

        # Ratings for various spam conditions
        raw_scores = []  # raw_scores - list of ratings
        if contains_stop_word(
            message
        ):  # if there is a stop word in the message, then add 0.90 to the list of ratings
            raw_scores.append(0.90)
        if contains_link(
            message
        ):  # if there is a link in the message, then add 0.05 to the list of ratings
            raw_scores.append(0.05)
        if contains_image(
            message
        ):  # if there is a photo in the message, then add 0.05 to the list of ratings
            raw_scores.append(0.05)
        if contains_mixed_alphabet(
            message
        ):  # if there are Cyrillic and Latin letters in one word, then add 0.05 to the list of ratings
            raw_scores.append(0.05)

        # Normalization of the score
        score = round(
            sum(raw_scores) / 1.05, 2
        )  # score - total score, sum(raw_scores) - sum of ratings, 1.05 - maximum possible score
        logger.info(f"Score: {score}")
        # If the score is greater than or equal to 0.90, then the message is sent to the administrator
        if score >= 0.90:
            logger.info(f"The message suspected of spam was sent to the administrator")
            await bot.send_message(
                ADMIN_ID,
                f'A message suspected of spam with a probability of {int(score * 100)} percent -->"{message.text}" <-- from user {message.from_user.id}',
            )
