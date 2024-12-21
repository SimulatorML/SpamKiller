import pandas as pd
from html import escape
from typing import Tuple, List
from aiogram import types, Bot
from loguru import logger

def extract_entities(message: types.Message) -> Tuple[str, str]:
    spoiler_link = ""
    if message.entities:
        for entity in message.entities:
            if entity.type == "text_link":
                spoiler_link = " " + entity.url + " " or ""
                break

    hidden_link = ""
    if message.caption_entities:
        for entity in message.caption_entities:
            if entity.type == "text_link":
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
    text = X.iloc[0, :].text
    user_id = X.iloc[0, :].from_id
    score = 0.0

    msg_features = {"label": None, "reasons": None, "model_name": "None",
                    "score": 0.0, "time_spent": 0.0, "prompt_name": "None",
                    "prompt_tokens": 0, 'completion_tokens': 0
                    }

    if user_id in admins or user_id in WHITELIST_ADMINS:
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: Админов нельзя трогать. Они хорошие"
        return msg_features

    if user_id in WHITELIST_USERS:
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: Пользователь в белом списке"
        return msg_features

    if not text:
        msg_features["label"] = 0
        msg_features["reasons"] = "Пояснение: нет текста в сообщении"
        return msg_features

    msg_features["model_name"] = "GptSpamClassifier"
    response = await gpt_classifier.predict(X)
    response = response[0]
    logger.info(response)
    keys = ['label', 'reasons', 'prompt_tokens', 'completion_tokens', 'time_spent', 'prompt_name']
    for key, value in zip(keys, response.values()):
        msg_features[key] = value

    if msg_features['label'] is None:
        msg_features['model_name'] = "RuleBasedClassifier"
        score, reasons = rule_based_classifier.predict(X)
        msg_features['score'] = score
        msg_features['reasons'] = "Причины:\n" + reasons
        
        # Новая логика для трехуровневой классификации
        if score == 2:  # Точно спам
            msg_features['label'] = 2
        elif score == 1:  # Возможно спам
            msg_features['label'] = 1
        else:  # Не спам
            msg_features['label'] = 0

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
) -> None:
    # Обрезаем длинный текст
    TRUNCATE_LENGTH = 600
    if len(text) > TRUNCATE_LENGTH:
        text = text[:TRUNCATE_LENGTH] + "..."

    # Формируем текст оценки только для rule-based классификатора
    score_text = (
        f"<b>({round(score * 100, 2)}%)</b>"
        if model_name == "RuleBasedClassifier"
        else ""
    )

    # Формируем базовое сообщение
    spam_message = _build_spam_message(
        message=message,
        text=text,
        user_description=user_description,
        model_name=model_name,
        prompt_name=prompt_name,
        time_spent=time_spent,
        reasons=reasons,
    )

    # Добавляем информацию о токенах для GPT
    if model_name == 'GptSpamClassifier':
        spam_message += (
            f"\n\nPrompt tokens: {prompt_tokens}\n"
            f"Completion tokens: {completion_tokens}"
        )

    try:
        await _handle_spam_actions(
            bot=bot,
            message=message,
            label=label,
            score=score,
            spam_message=spam_message,
            photo=photo,
            GROUP_CHAT_ID=GROUP_CHAT_ID,
            ADMIN_IDS=ADMIN_IDS,
            TARGET_SPAM_ID=TARGET_SPAM_ID,
            TARGET_NOT_SPAM_ID=TARGET_NOT_SPAM_ID,
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке спам-сообщения: {e}")

def _build_spam_message(
    message: types.Message,
    text: str,
    user_description: str,
    model_name: str,
    prompt_name: str,
    time_spent: float,
    reasons: str,
) -> str:
    """Формирует текст сообщения о спаме."""
    return (
        f"Канал: <a href='t.me/{message.chat.username}'>{message.chat.title}</a>\n"
        f"Автор: @{message.from_user.username}\n"
        f"User_id: {message.from_user.id}\n"
        f"Время: {message.date}\n\n"
        "Текст сообщения:\n"
        '"""\n'
        f"{escape(text)}\n"
        '"""\n\n'
        "Описание аккаунта:\n"
        f"{'-' * 10}\n"
        f"{escape(user_description)}\n"
        f"{'-' * 10}\n\n"
        f"Model: {model_name}\n"
        f"Prompt: {prompt_name}\n"
        f"Time spent for prediction: {time_spent} seconds\n\n"
        f"{reasons}"
    )

async def _handle_spam_actions(
    bot: Bot,
    message: types.Message,
    label: int,
    score: float,
    spam_message: str,
    photo,
    GROUP_CHAT_ID: int,
    ADMIN_IDS: List[int],
    TARGET_SPAM_ID: int,
    TARGET_NOT_SPAM_ID: int,
) -> None:
    """Обрабатывает действия со спам-сообщением."""
    
    async def send_message_or_photo(target_id: int) -> None:
        if photo:
            await bot.send_photo(
                chat_id=target_id,
                photo=photo,
                caption=spam_message
            )
        else:
            await bot.send_message(
                chat_id=target_id,
                text=spam_message
            )

    # Обработка спама
    if label >= 1:
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id
        )
        await send_message_or_photo(target_id=GROUP_CHAT_ID)
        await bot.ban_chat_member(
            chat_id=GROUP_CHAT_ID,
            user_id=message.from_user.id
        )
    
    # Обработка подозрительных сообщений
    elif score > 0.3:
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id
        )

    # Отправка уведомлений
    for admin_id in ADMIN_IDS:
        await send_message_or_photo(target_id=admin_id)

    # Отправка в целевой канал
    target_id = TARGET_SPAM_ID if label >= 1 else TARGET_NOT_SPAM_ID
    await send_message_or_photo(target_id=target_id)

