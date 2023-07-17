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
    if True: #await check_user_id(message):
        logger.info(f"Message got from new user. Checking for spam")
        X = pd.DataFrame({'text': [message.text], 'photo': [message.photo]})   
        scores = classifier.predict(X)  # Wrap the message in a list
        score = scores[0]  # Extract the score from the list
        logger.info(f"Score: {score}")

        if score >= 0.90:
            logger.info(
                f"The message suspected of spam was sent to the administrator and the group"
            )
            spam_message_for_admins = f'A message suspected of spam with a probability of {int(score * 100)} percent -->"{message.text}" <-- from user {message.from_user.id}. Lable is 1'
            spam_message_for_group = f"A message suspected of spam with a probability of {int(score * 100)} percent --> {message.text} <--. Lable is 1"
            for admin_id in ADMIN_IDS:
                await bot.send_message(admin_id, spam_message_for_admins)
            # Send the same message to the group
            await bot.send_message(GROUP_CHAT_ID, spam_message_for_group)
        else:
            logger.info(
                f"The message is not suspected of spam. Sent to the administrator and the group"
            )
            is_not_span_message = f'This message is not suspected of spam with a probability of {int(score * 100)} percent -->"{message.text}" <-- from user {message.from_user.id}. Lable is 0'
            await bot.send_message(GROUP_CHAT_ID, is_not_span_message)
