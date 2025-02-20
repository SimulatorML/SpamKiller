from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger
from src.config import (
    WHITELIST_USERS,
    GOLDLIST_USERS,
)
from src.utils.commands import add_user_to_whitelist
from src.utils.message_processing import (
    extract_entities,
    build_data_frame,
    classify_message,
    send_spam_alert,
)

# Константы
DAYS_WINDOW = 7 # порог дней для бана, если метка == likely spam шла 2 раза подряд за период n дней
N_LIKELY_SPAM_MESS = 2 # количество меток == likely spam, при котором будет бан за период n дней
MESSAGES_TO_WHITELIST = 3 # порог не спам сообщений для белого листа

# Хранение состояния сообщений пользователей с временными метками
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
    logger.info(f"Starting message processing for user {message.from_id}")

    try:
        # Обработка форвард-сообщений и историй
        forward_info = ""
        is_story = False
        story_caption = ""
        
        # Определяем photo в начале функции
        photo = None
        if message.photo:
            photo = message.photo[-1].file_id
        
        if message.forward_from or message.forward_from_chat:
            if message.forward_from:
                forward_info = f"Forwarded from user @{message.forward_from.username or 'Unknown'} (ID: {message.forward_from.id})"
            elif message.forward_from_chat:
                forward_info = f"Forwarded from chat {message.forward_from_chat.title} (ID: {message.forward_from_chat.id})"
            
            # Улучшенная проверка на историю
            if getattr(message, 'forward_from_story', None):
                is_story = True
                forward_info += " [STORY]"
                
                # Получаем caption истории
                if hasattr(message.forward_from_story, 'caption'):
                    story_caption = message.forward_from_story.caption or ""
                # Дополнительно проверяем caption в самом сообщении
                elif message.caption:
                    story_caption = message.caption
                
                logger.info(f"Story detected. Caption: {story_caption}")
        
        # Добавляем информацию о форварде и caption истории к тексту для анализа
        original_text = message.text or message.caption or ""
        text = original_text
        
        if forward_info:
            text += f"\n[{forward_info}]"
        
        if story_caption:
            text += f"\n[Story caption: {story_caption}]"
            logger.info(f"Final text with story caption: {text}")

        # Сбор данных
        media_type = None
        media_id = None
        
        if message.photo:
            media_type = "photo"
            media_id = message.photo[-1].file_id
        elif message.video:
            media_type = "video"
            media_id = message.video.file_id
        
        # Получение информации о пользователе
        try:
            user_info = await bot.get_chat(message.from_user.id)
            user_description = user_info.bio or ""
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            user_description = ""

        # Добавляем дополнительные признаки для форвард-сообщений и историй
        X = build_data_frame(
            text=text,
            bio=user_description,
            from_id=message.from_id,
            media_type=media_type,
            media_id=media_id,
            story_caption=story_caption,
            reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
            channel=message.chat.username,
            is_forwarded=bool(message.forward_from or message.forward_from_chat),
            is_story=is_story,
            forward_from_id=message.forward_from.id if message.forward_from else None,
            forward_from_chat_id=message.forward_from_chat.id if message.forward_from_chat else None,
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
            GOLDLIST_USERS=GOLDLIST_USERS,
            bot=bot,
            chat_id=GROUP_CHAT_ID,
        )

        # Добавление сообщения в буфер пользователя с временной меткой
        current_time = datetime.now()
        user_message_buffer[message.from_id].append({
            "label": msg_features["label"],
            "timestamp": current_time
        })

        # Проверка на добавление в белый список (3 не-спам сообщения, MESSAGES_TO_WHITELIST не спам-сообщений)
        messages = user_message_buffer[message.from_id]
        if len(messages) >= MESSAGES_TO_WHITELIST:
            last_messages = messages[-MESSAGES_TO_WHITELIST:]
            if all(msg["label"] == 0 for msg in last_messages):
                if message.from_id not in WHITELIST_USERS:
                    WHITELIST_USERS.append(message.from_id)
                    add_user_to_whitelist(user_id=message.from_id)
                    # Очищаем буфер после добавления в белый список
                    user_message_buffer[message.from_id].clear()

                    logger.info(f"User {message.from_id} added to whitelist after {MESSAGES_TO_WHITELIST} non-spam messages.")

        # Проверка на спам (любые два label==1 в течение DAYS_WINDOW дней)
        if len(messages) >= N_LIKELY_SPAM_MESS:
            # Получаем все сообщения с label==1 за последние DAYS_WINDOW дней
            spam_messages = [msg for msg in messages if msg["label"] == 1]

            if len(spam_messages) >= N_LIKELY_SPAM_MESS:
                # Проверяем только последние соседние спам-сообщения
                time_diff = (spam_messages[-1]["timestamp"] - spam_messages[-2]["timestamp"]).days
                if time_diff <= DAYS_WINDOW:
                    logger.warning(f"User {message.from_id} has sent two spam messages within {time_diff} days")
                    # Повышаем уровень угрозы
                    msg_features["label"] = 2
                    msg_features["reasons"] += f"\nПовышен уровень угрозы: найдено 2 спам-сообщения за {time_diff} дней"
                    # Очищаем буфер после бана
                    user_message_buffer[message.from_id].clear()
                    try:
                        await bot.ban_chat_member(
                            chat_id=message.chat.id,
                            user_id=message.from_id
                        )

                        logger.info(f"Banned user {message.from_id} for repeated spam messages")
                    except Exception as e:
                        logger.error(f"Failed to ban user {message.from_id}: {e}")

        # Добавим логирование перед отправкой сообщения
        logger.info(f"Preparing to send spam alert. Label: {msg_features['label']}")
        
        # Проверим все необходимые параметры перед отправкой
        logger.info(f"Message features: {msg_features}")
        
        await send_spam_alert(
            bot=bot,
            message=message,
            label=msg_features["label"],
            reasons=msg_features["reasons"],
            text=text,
            prompt_name=msg_features.get("prompt_name", "None"),  # Используем .get() с дефолтными значениями
            model_name=msg_features.get("model_name", "None"),
            score=msg_features.get("score", 0.0),
            time_spent=msg_features.get("time_spent", 0.0),
            prompt_tokens=msg_features.get("prompt_tokens", 0),
            completion_tokens=msg_features.get("completion_tokens", 0),
            photo=photo,  # Передаем определенную переменную photo
            user_description=user_description,
            GROUP_CHAT_ID=GROUP_CHAT_ID,
            ADMIN_IDS=ADMIN_IDS,
            TARGET_SPAM_ID=TARGET_SPAM_ID,
            TARGET_NOT_SPAM_ID=TARGET_NOT_SPAM_ID,
            WHITELIST_USERS=WHITELIST_USERS,
            profile_analysis=msg_features.get("profile_analysis"),
        )
        
        logger.info("Spam alert sent successfully")
        
    except Exception as e:
        logger.error(f"Error in handle_msg_with_args: {e}")
        # Можно добавить отправку уведомления об ошибке администраторам
        try:
            error_message = f"Error processing message:\n{str(e)}"
            await bot.send_message(GROUP_CHAT_ID, error_message)
        except:
            pass
    
