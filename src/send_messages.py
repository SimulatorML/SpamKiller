import os
from loguru import logger
from aiogram import types
import pandas as pd
from src.add_new_user_id import check_user_id


# Reading only new messages from new users
async def handle_msg_with_args(
    message: types.Message, bot, classifier, ADMIN_IDS, GROUP_CHAT_ID
):
    """
    Function for processing messages from users and sending them to the administrator if the message is suspected of spam

    Parameters
    ----------
    message : types.Message
        Message from user
    bot : Bot
        Bot
    ADMIN_ID : str
        Admin id

    Returns
    -------
    None
    """

    if True:  # await check_user_id(message):
        logger.info(f"Message got from new user. Checking for spam")
        X = pd.DataFrame(
            {
                "text": [message.text],
                "photo": "photo" in message,
                "from_id": message.from_id,
                "reply_to_message_id": "reply_to_message_id" in message,
            }

        scores = classifier.predict(X)  # Wrap the message in a list
        score = scores[0]  # Extract the score from the list
        logger.info(f"Score: {score}")

        treshold = 0.90
        if score >= treshold:
            label = "Spam"
            reason = f"score >= {treshold}"
        else:
            label = "Not spam"
            reason = f"score < {treshold}"

        logger.info(f"The message was sent to the administrator and the group")

        spam_message_for_admins = (
            f"Канал: {message.chat.title}\n"
            + f"Автор: {message.from_id}\n"
            + f"Время: {message.date}\n"
            + f"\n"
            + f"{message.text}\n"
            + f"\n"
            + f"{label}\n"
            + f"Оценка: {score}\n"
            + f"Причина: {reason}"
        )

        spam_message_for_group = spam_message_for_admins
        for admin_id in ADMIN_IDS:
            await bot.send_message(admin_id, spam_message_for_admins)
        # Send the same message to the group
        await bot.send_message(GROUP_CHAT_ID, spam_message_for_group)


# spam_message_for_admins = f"Канал: {message.chat.title} \nАвтор: {message.from_id} \nВремя: {message.date} \n \n{message.text} \n \n{label} \nОценка: {score} \nПричина: {reason}"
