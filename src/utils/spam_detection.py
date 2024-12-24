from collections import defaultdict
from loguru import logger
from src.config import WHITELIST_USERS
from src.utils.commands import add_user_to_whitelist
from src.utils.message_processing import (
    extract_entities,
    build_data_frame,
    classify_message,
    send_spam_alert,
)

# Хранение состояния сообщений пользователей
user_message_buffer = defaultdict(list)

async def handle_msg_with_args(
    message,
    bot,
    gpt_classifier,
    rule_based_classifier,
    THRESHOLD_RULE_BASED,
    ADMIN_IDS,
    GROUP_CHAT_ID,
    AUTHORIZED_USER_IDS,
    AUTHORIZED_GROUP_IDS,
    TARGET_SPAM_ID,
    TARGET_NOT_SPAM_ID,
    WHITELIST_ADMINS,
):
    logger.info(
        f"Got new message from a user {message.from_id} in {message.chat.id} ({message.chat.username}). Checking for spam..."
    )

    # Сбор данных
    photo = message.photo[-1].file_id if message.photo else None
    text = message.text or message.caption or ""
    user_info = await bot.get_chat(message.from_user.id)
    user_description = user_info.bio or ""
    reply_to_message_id = (
        message.reply_to_message.message_id if message.reply_to_message else None
    )
    channel = message.chat.username

    spoiler_link, hidden_link = extract_entities(message=message)

    text = text[:550]
    text += spoiler_link
    text += hidden_link

    X = build_data_frame(
        text=text,
        bio=user_description,
        from_id=message.from_id,
        photo=photo,
        reply_to_message_id=reply_to_message_id,
        channel=channel,
    )

    channel_admins_info = await bot.get_chat_administrators(message.chat.id)
    admins = [admin.user.id for admin in channel_admins_info]

    # Классификация сообщения
    msg_features = await classify_message(
        X=X,
        gpt_classifier=gpt_classifier,
        rule_based_classifier=rule_based_classifier,
        THRESHOLD_RULE_BASED=THRESHOLD_RULE_BASED,
        admins=admins,
        WHITELIST_ADMINS=WHITELIST_ADMINS,
        WHITELIST_USERS=WHITELIST_USERS,
    )

    # Добавление сообщения в буфер пользователя
    user_message_buffer[message.from_id].append(msg_features["label"])

    # Проверка состояния буфера
    if len(user_message_buffer[message.from_id]) == 3:  # Если пользователь отправил 3 сообщения
        if all(label == 0 for label in user_message_buffer[message.from_id]):
            # Если все 3 сообщения не являются спамом
            if message.from_id not in WHITELIST_USERS:
                WHITELIST_USERS.append(message.from_id)
                add_user_to_whitelist(user_id=message.from_id)
                logger.info(f"User {message.from_id} added to whitelist after 3 non-spam messages.")
        # Очистка буфера после обработки 3 сообщений
        del user_message_buffer[message.from_id]
        
         # Отправка уведомления (для администраторов или журналов)
    await send_spam_alert(
        bot=bot,
        message=message,
        label=msg_features["label"],
        reasons=msg_features["reasons"],
        text=text,
        prompt_name=msg_features["prompt_name"],
        model_name=msg_features["model_name"],
        score=msg_features["score"],
        time_spent=msg_features["time_spent"],
        prompt_tokens=msg_features["prompt_tokens"],
        completion_tokens=msg_features["completion_tokens"],
        photo=photo,
        user_description=user_description,
        GROUP_CHAT_ID=GROUP_CHAT_ID,
        ADMIN_IDS=ADMIN_IDS,
        TARGET_SPAM_ID=TARGET_SPAM_ID,
        TARGET_NOT_SPAM_ID=TARGET_NOT_SPAM_ID,
    )
