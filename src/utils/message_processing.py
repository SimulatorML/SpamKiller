import pandas as pd
from html import escape
from typing import Tuple, List
from aiogram import types, Bot
from loguru import logger


def extract_entities(message: types.Message) -> Tuple[str, str]:
    spoiler_link = ""
    if message.entities:  # Проверка на наличие сущностей в сообщении
        for entity in message.entities:
            if entity.type == "text_link":  # Если сущность является текстовой ссылкой
                spoiler_link = " " + entity.url + " " or ""
                break

    hidden_link = ""
    if message.caption_entities:  # Проверка для скрытых ссылок в тексте в сообщении
        for entity in message.caption_entities:
            if entity.type == "text_link":  # Если сущность является текстовой ссылкой
                hidden_link = " " + entity.url + " " or ""

    return spoiler_link, hidden_link


def build_data_frame(
    text: str, bio: str, from_id: int, photo, reply_to_message_id, channel: str
) -> pd.DataFrame:
    data = {
        "text": [text],
        "bio": [bio],
        "from_id": [from_id],
        "photo": [photo],
        "reply_to_message_id": [reply_to_message_id],
        "channel": [channel],
    }
    data = pd.DataFrame(data)
    return data


async def classify_message(
    X: pd.DataFrame,
    gpt_classifier,
    rule_based_classifier,
    THRESHOLD_RULE_BASED: float,
    admins: List[int],
    WHITELIST_ADMINS: List[int],
    WHITELIST_USERS: List[int],
) -> dict:
    # Get user_id
    text = X.iloc[0, :].text
    user_id = X.iloc[0, :].from_id
    score = 0.0

    msg_features = {"label": None, "reasons": None, "model_name": "None",
                    "score": 0.0, "time_spent": 0.0, "prompt_name": "None",
                    "prompt_tokens": 0, 'completion_tokens': 0
                    }

    # Check for special user categories
    if user_id in admins or user_id in WHITELIST_ADMINS:
        # If the user is admin
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: Админов нельзя трогать. Они хорошие"
        return msg_features

    if user_id in WHITELIST_USERS:
        # If the user is in whitelist users
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: Пользователь в белом списке"
        return msg_features

    if not text:
        # If the message doesn't contain any text
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: нет текста в сообщении"
        return msg_features

    # Classifying the message
    msg_features["model_name"] = "GptSpamClassifier"
    response = await gpt_classifier.predict(X)
    response = response[0]
    logger.info(response)
    keys = ['label', 'reasons', 'prompt_tokens', 'completion_tokens', 'time_spent', 'prompt_name']
    for key, value in zip(keys, response.values()):
        msg_features[key] = value

    # If there was an Error with OpenAI (timeout, unexpected response or different error), rule_based model will be used
    if msg_features['label'] is None:
        msg_features['model_name'] = "RuleBasedClassifier"
        msg_features['score'], msg_features['reasons'] = rule_based_classifier.predict(X)

        msg_features['reasons'] = "Причины:\n" + msg_features['reasons']
        msg_features['label'] = 1 if score >= THRESHOLD_RULE_BASED else 0

    return msg_features




async def send_spam_alert(
    bot: Bot,
    message: types.Message,
    label: int,
    reasons: str,
    text: str,
    prompt_name: str,
    model_name: str,
    score: float,
    time_spent: float,
    prompt_tokens: int,
    completion_tokens: int,
    photo,
    user_description: str,
    GROUP_CHAT_ID: int,
    ADMIN_IDS: List[int],
    TARGET_SPAM_ID: int,
    TARGET_NOT_SPAM_ID: int,
):
    if label == 1:
        label_text = "<b>&#8252;&#65039; Spam DETECTED</b>"
    else:
        label_text = "<i>No spam detected</i>"
    if len(text) > 600:
        text = text[:600] + "..."

    score_text = (
        f"<b>({round(score * 100, 2)}%)</b>"
        if model_name == "RuleBasedClassifier"
        else ""
    )

    logger.info("The message was sent to the administrator and the group")

    spam_message = (
        f"{label_text} {score_text}"
        + "\n"
        + f"Канал: <a href='t.me/{message.chat.username}'>{message.chat.title}</a>\n"  # <a href="url">link text</a>
        + f"Автор: @{(message.from_user.username)}\n"
        + f"User_id: {(message.from_user.id)}\n"
        + f"Время: {(message.date)}\n\n"
        + "Текст сообщения:\n"
        + '"""\n'
        + f"{escape(text)}\n"
        + '"""\n\n'
        + "Описание аккаунта:\n"
        + "-" * 10
        + "\n"
        + f"{escape(user_description)}\n"
        + "-" * 10
        + "\n"
        + "\n"
        + f"Model: {model_name}\n"
        + f"Prompt: {prompt_name}\n"
        + f"Time spent for prediction: {time_spent} seconds\n"
        + "\n"
        + reasons
    )
    if model_name == 'GptSpamClassifier':
        spam_message += (f"\n\nP_tokens: {prompt_tokens}\n"
                         + f"C_tokens: {completion_tokens}")

    async def send_message_or_photo(target_id, spam_message=spam_message, photo=photo):
        """"""
        if photo:
            await bot.send_photo(chat_id=target_id, photo=photo, caption=spam_message)
        else:
            await bot.send_message(chat_id=target_id, text=spam_message)

    if label == 1:
        await send_message_or_photo(target_id=GROUP_CHAT_ID)

    for admin_id in ADMIN_IDS:
        await send_message_or_photo(target_id=admin_id)

    target_id = TARGET_SPAM_ID if label == 1 else TARGET_NOT_SPAM_ID
    await send_message_or_photo(target_id=target_id)

    if label == 1:
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
