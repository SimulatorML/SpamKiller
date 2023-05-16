import pandas as pd
from loguru import logger
from aiogram import types
import re
from fuzzywuzzy import fuzz

logger.info("Обработка сообщений")

df = pd.read_csv("data/text_spam_dataset/cleaned_spam.csv", sep=";")
STOP_WORDS = df["text"].astype(str).tolist()  # Переводим все элементы в строки


def contains_stop_word(message):
    logger.info(f"Проверка на наличие стоп-слова в сообщении: {message}")
    if message is None:  # Проверка на существование message.text
        return False
    for word in STOP_WORDS:  # STOP_WORDS - список стоп-слов
        if (
            fuzz.ratio(message, word) >= 80
        ):  # fuzz.ratio() возвращает схожесть строк в процентах
            return True
    return False


# Проверка на наличие ссылки в сообщении
def contains_link(message):  # message - объект сообщения
    logger.info(f"Проверка на наличие ссылки в сообщении")
    return (
        "https://t.me" in message or "t.me" in message or "https://" in message
    )  # message - объект сообщения


# Проверка на наличие кириллицы и латиницы в одном слове
def contains_mixed_alphabet(message):  # message - объект сообщения
    logger.info(f"Проверка на наличие кириллицы и латиницы в одном слове")
    text = message  # message.text - текст сообщения
    if re.search("[а-я]", text) and re.search(
        "[a-z]", text
    ):  # re.search() - поиск по регулярному выражению для поиска кириллицы и латиницы в слове
        return True
    return False


# Обработка сообщения
def handle_msg_with_args(message):  # Проверка на наличие пользователя в базе данных
    logger.info(f"Получено сообщение: {message}")

    # Оценки для различных спам-условий
    raw_scores = []  # raw_scores - список оценок
    if contains_stop_word(
        message
    ):  # если в сообщении есть стоп-слово, то добавляем в список оценок 0.90
        raw_scores.append(0.90)
    if contains_link(
        message
    ):  # если в сообщении есть ссылка, то добавляем в список оценок 0.05
        raw_scores.append(0.05)
    if contains_mixed_alphabet(
        message
    ):  # если в сообщении есть кириллица и латиница в одном слове, то добавляем в список оценок 0.05
        raw_scores.append(0.05)

    # Нормализация счета
    score = round(
        sum(raw_scores) / 1, 2
    )  # score - общий балл, sum(raw_scores) - сумма оценок, 1.05 - максимальный балл

    # Если общий балл превышает порог, отправьте сообщение администратору
    if score >= 0.90:
        logger.info(f"Сообщение, подозреваемое в спаме отправлено администратору")
        print(
            f"Сообщение, подозреваемое в спаме с вероятностью {int(score*100)} процентов отправлено администратору"
        )


# Тесты
handle_msg_with_args("Все вещи ДАРОМ с ВБ уже тут - https://tel")
