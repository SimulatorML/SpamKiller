import pandas as pd
from loguru import logger
import re
from fuzzywuzzy import fuzz
import yaml

# Read data
with open("./config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    cleaned_spam_path = config["cleaned_spam"]
    clened_not_spam_path = config["clened_not_spam"]
    df_cleaned_spam_and_not_spam_path = config["df_cleaned_spam_and_not_spam"]

# Load data cleaned_spam from CSV
cleaned_spam = pd.read_csv(cleaned_spam_path, sep=";")

# Load data clened_not_spam from CSV
clened_not_spam = pd.read_csv(clened_not_spam_path, sep=";")
# # Concatenating two dataframes

df_cleaned_spam_and_not_spam = pd.concat(
    [cleaned_spam, clened_not_spam], ignore_index=True
)

example_spam = cleaned_spam["text"].astype(str).tolist()

# Create a list of spam words
spam_words = []
for text in example_spam:
    spam_words.extend(text.split())


# Check if message contains stop words
def contains_stop_word(message):
    if message is None:
        return False
    return message in example_spam


# Check if message contains stop words with fuzzywuzzy
# def contains_stop_word(message):
#     if message is None:
#         return False
#     for spam_message in example_spam:
#         if fuzz.ratio(message, spam_message) >= 80:
#             return True
#     return False


# Check if message contains link
def contains_link(message):
    return "https://t.me" in message or "t.me" in message or "https://" in message


# Check if message contains mixed alphabet
def contains_mixed_alphabet(message):
    text = message
    if re.search("[а-я]", text) and re.search("[a-z]", text):
        return True
    return False


# Check if message contains spam criteria
def handle_msg_with_args(message):
    raw_scores = []
    if contains_stop_word(message):
        raw_scores.append(0.90)
    if contains_link(message):
        raw_scores.append(0.05)
    if contains_mixed_alphabet(message):
        raw_scores.append(0.05)

    score = round(sum(raw_scores) / 1, 2)
    return score


pred_scores = [
    handle_msg_with_args(msg) for msg in df_cleaned_spam_and_not_spam["text"]
]
df_cleaned_spam_and_not_spam["pred_scores"] = pred_scores
df_cleaned_spam_and_not_spam.to_csv(
    df_cleaned_spam_and_not_spam_path, sep=";", index=False
)
true_labels = df_cleaned_spam_and_not_spam["label"].values
