import pandas as pd
from html import escape
from typing import Tuple, List
from aiogram import types, Bot
from loguru import logger
import asyncio
from src.models.user_analisys import ProfileClassifier
from datetime import datetime, timedelta

# Добавить глобальные переменные для отслеживания состояния
profile_analyzer_active = False
profile_analyzer_start_time = None
ANALYSIS_DURATION = timedelta(minutes=5)

# Добавить функцию активации анализатора
async def activate_profile_analyzer():
    global profile_analyzer_active, profile_analyzer_start_time
    profile_analyzer_active = True
    profile_analyzer_start_time = datetime.now()
    logger.info("Profile analyzer activated for 5 minutes")

# Добавить функцию проверки состояния анализатора
def is_profile_analyzer_active():
    global profile_analyzer_active, profile_analyzer_start_time
    if not profile_analyzer_active:
        return False
    
    if datetime.now() - profile_analyzer_start_time > ANALYSIS_DURATION:
        profile_analyzer_active = False
        logger.info("Profile analyzer deactivated due to timeout")
        return False
    
    return True

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
    GOLDLIST_USERS: List[int],
) -> dict:
    text = X.iloc[0, :].text
    user_id = X.iloc[0, :].from_id
    score = 0.0

    msg_features = {
        "label": None, "reasons": None, "model_name": "None",
        "score": 0.0, "time_spent": 0.0, "prompt_name": "None",
        "prompt_tokens": 0, 'completion_tokens': 0,
        "profile_analysis": None
    }

    # Проверяем, нужно ли анализировать профиль
    profile_analysis_enabled = is_profile_analyzer_active()
    profile_classifier = None

    try:
        # Проверяем активацию анализатора только для goldlist
        if user_id in GOLDLIST_USERS:
            # Если сообщение от пользователя из goldlist, активируем анализатор
            await activate_profile_analyzer()
            logger.info(f"Profile analyzer activated by goldlist user {user_id}")

        # Проверяем белые списки
        if user_id in admins:
            msg_features["label"] = 0
            msg_features["reasons"] = "Пояснение: Админов нельзя трогать. Они хорошие"
            return msg_features

        if user_id in WHITELIST_ADMINS or user_id in WHITELIST_USERS:
            msg_features["label"] = 0
            msg_features["reasons"] = "Пояснение: Пользователь в белом списке"
            return msg_features

        if not text:
            msg_features["label"] = 0
            msg_features["reasons"] = "Пояснение: нет текста в сообщении"
            return msg_features

        # Получаем результаты анализа сообщения
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
            
            if score == 2:
                msg_features['label'] = 2
            elif score == 1:
                msg_features['label'] = 1
            else:
                msg_features['label'] = 0

        # Анализируем профиль только если анализатор активен
        if profile_analysis_enabled:
            try:
                profile_classifier = ProfileClassifier()
                profile_results = await profile_classifier.analyze_profile(user_id)
                logger.info(f"Profile analysis results: {profile_results}")
                
                msg_features["profile_analysis"] = profile_results
                
                # Повышаем уровень угрозы, если профиль подозрительный
                if profile_results.get('overall_score', 0) >= 0.8 and msg_features['label'] < 2:
                    msg_features['label'] = min(msg_features['label'] + 1, 2)
                    msg_features['reasons'] += "\n\n⚠️ Уровень угрозы повышен из-за подозрительного профиля"
                
                elif profile_results.get('overall_score', 0) == 0 and 'особенности' in profile_results.get('features', '').lower():
                    msg_features['label'] = max(msg_features['label'], 1)
                    msg_features['reasons'] += "\n\n⚠️ Подозрительные особенности в профиле, несмотря на низкую оценку"
            
            except Exception as e:
                logger.error(f"Error in profile analysis: {e}")

    except Exception as e:
        logger.error(f"Error in classify_message: {e}")
        return {
            "label": 0,
            "reasons": f"Error during message classification: {str(e)}",
            "model_name": "Error",
            "score": 0.0,
            "time_spent": 0.0,
            "prompt_name": "None",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "profile_analysis": None
        }
    finally:
        # Закрываем клиент в блоке finally
        if profile_classifier:
            try:
                await profile_classifier.close()
            except Exception as e:
                logger.error(f"Error closing profile classifier: {e}")

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
    WHITELIST_USERS: List[int] = None,
    profile_analysis: dict = None,  # Добавляем параметр для анализа профиля
):
    """
    Отправляет уведомление о спаме и удаляет спам-сообщения.
    Спам-сообщения удаляются для ВСЕХ пользователей, включая тех, кто в белом списке.
    """
    try:
        # Добавим логирование в начале функции
        logger.info(f"Starting send_spam_alert for message {message.message_id}")
        
        # Проверяем, является ли сообщение спамом (label >= 1)
        if label >= 1:
            try:
                # Проверяем права бота
                bot_member = await bot.get_chat_member(message.chat.id, bot.id)
                if not bot_member.can_delete_messages:
                    logger.error(f"Bot doesn't have permission to delete messages in chat {message.chat.id}")
                    return

                # Удаляем сообщение независимо от статуса пользователя
                await bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=message.message_id
                )

                # Логируем с учетом статуса пользователя
                is_whitelisted = WHITELIST_USERS and message.from_user.id in WHITELIST_USERS
                
                if is_whitelisted:
                    logger.warning(
                        f"Deleted spam message {message.message_id} from whitelisted user {message.from_user.id}. "
                        f"Spam score: {score}, Label: {label}"
                    )
                else:
                    logger.info(
                        f"Deleted spam message {message.message_id} from user {message.from_user.id}. "
                        f"Spam score: {score}, Label: {label}"
                    )

                    # Баним только если:
                    # 1. Сообщение точно спам (label == 2)
                    # 2. Пользователь точно НЕ в белом списке
                    if label == 2:
                        try:
                            await bot.ban_chat_member(
                                chat_id=message.chat.id,
                                user_id=message.from_user.id
                            )
                            logger.info(f"Banned user {message.from_user.id} for spam")
                        except Exception as e:
                            logger.error(f"Failed to ban user {message.from_user.id}: {e}")

            except Exception as e:
                logger.error(f"Failed to process spam message {message.message_id}: {e}")

        # Формируем сообщение для отправки с учетом анализа профиля
        spam_message = _build_spam_message(
            message=message,
            label=label,
            text=text,
            user_description=user_description,
            model_name=model_name,
            prompt_name=prompt_name,
            time_spent=time_spent,
            reasons=reasons,
            score=score,
            is_whitelisted=WHITELIST_USERS and message.from_user.id in WHITELIST_USERS,
            profile_analysis=profile_analysis  # Передаем результаты анализа профиля
        )

        logger.info(f"Built spam message: {spam_message[:100]}...")  # Логируем начало сообщения
        
        # Добавляем информацию о токенах для GPT
        if model_name == 'GptSpamClassifier':
            spam_message += (
                f"\n\nPrompt tokens: {prompt_tokens}\n"
                f"Completion tokens: {completion_tokens}"
            )

        # Отправляем уведомления
        sent_to = set()  # Множество для отслеживания ID, куда уже отправили сообщение
        tasks = []

        # В общий чат отправляем всегда
        if GROUP_CHAT_ID:  # Проверяем, что ID существует
            tasks.append(send_message_or_photo(bot, GROUP_CHAT_ID, spam_message, photo))
            sent_to.add(GROUP_CHAT_ID)
            logger.info(f"Added task to send to GROUP_CHAT_ID: {GROUP_CHAT_ID}")

        # Администраторам отправляем, если им ещё не отправляли
        for admin_id in ADMIN_IDS:
            if admin_id not in sent_to:
                tasks.append(send_message_or_photo(bot, admin_id, spam_message, photo))
                sent_to.add(admin_id)

        # Отправляем в соответствующий канал, если ещё не отправляли
        target_id = TARGET_SPAM_ID if label >= 1 else TARGET_NOT_SPAM_ID
        if target_id not in sent_to:
            tasks.append(send_message_or_photo(bot, target_id, spam_message, photo))

        # Выполняем все отправки асинхронно
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Error in send_spam_alert: {e}")
        # Пытаемся отправить базовое сообщение об ошибке
        try:
            await bot.send_message(
                GROUP_CHAT_ID,
                f"Error sending spam alert: {str(e)}\nMessage ID: {message.message_id}"
            )
        except:
            pass

async def send_message_or_photo(bot: Bot, chat_id: int, text: str, photo) -> None:
    """Вспомогательная функция для отправки сообщения с фото или без"""
    try:
        if photo:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=text,
                parse_mode="HTML"
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML"
            )
    except Exception as e:
        logger.error(f"Failed to send message to {chat_id}: {e}")

def _build_spam_message(
    message: types.Message,
    label: int,
    text: str,
    user_description: str,
    model_name: str,
    prompt_name: str,
    time_spent: float,
    reasons: str,
    score: float,
    is_whitelisted: bool = False,
    profile_analysis: dict = None,  # Добавляем параметр для анализа профиля
) -> str:
    """Формирует текст сообщения  учетом статуса пользователя"""
    
    # Определяем тип сообщения
    if label == 2:
        label_text = "🚨 <b>SPAM DETECTED (High confidence)</b>"
    elif label == 1:
        label_text = "⚠️ <b>Possible SPAM</b>"
    else:
        label_text = "✅ <i>No spam detected</i>"

    # Добавляем пометку о белом списке
    whitelist_status = "⭐️ [Whitelisted user]" if is_whitelisted else ""

    score_text = (
        f"<b>({round(score * 100, 2)}%)</b>"
        if model_name == "RuleBasedClassifier"
        else ""
    )

    # Обрезаем длинный текст
    MAX_TEXT_LENGTH = 600
    if text and len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "..."

    result = (
        f"{label_text} {score_text} {whitelist_status}\n"
        f"Канал: <a href='t.me/{message.chat.username}'>{message.chat.title}</a>\n"
        f"Автор: @{message.from_user.username or 'None'}\n"
        f"User_id: {message.from_user.id}\n"
        f"Время: {message.date}\n\n"
        f"Текст сообщения:\n"
        '"""\n'
        f"{escape(text)}\n"
        '"""\n\n'
        f"Описание аккаунта:\n"
        f"{'-' * 10}\n"
        f"{escape(user_description)}\n"
        f"{'-' * 10}\n\n"
    )

    # Добавляем результаты анализа профиля
    if profile_analysis:
        result += (
            f"Анализ профиля:\n"
            f"{'-' * 10}\n"
            f"Общая оценка: {profile_analysis.get('overall_score', 0):.2f}\n"
            f"Особенности:\n{profile_analysis.get('features', 'Нет данных')}\n"
            f"{'-' * 10}\n\n"
        )

    result += (
        f"Model: {model_name}\n"
        f"Prompt: {prompt_name}\n"
        f"Time spent: {time_spent:.2f} seconds\n\n"
        f"{reasons}"
    )

    return result

