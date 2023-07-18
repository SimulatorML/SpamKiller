import os
from loguru import logger
from aiogram import types
import pandas as pd
from src.add_new_user_id import check_user_id
from aiogram.types import ParseMode
from html import escape
# Reading only new messages from new users
async def handle_msg_with_args(message, bot, classifier, ADMIN_IDS, GROUP_CHAT_ID):
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

        try:
            reply_to_message_id = message.reply_to_message
        except:
            reply_to_message_id = None
        
        try:
            photo = message.photo
        except:
            photo = None

        if message.text or message.caption:
            text = message.text or message.caption
        else:
            text = ""

        print(photo)
        print(reply_to_message_id)
        print([text])
        print(message.from_id)
        X = {
            "text": text,
            "photo": photo,
            "from_id": message.from_id,
            "reply_to_message_id": reply_to_message_id,
        }
        print(X)
        scores = classifier.predict(X)  # Wrap the message in a list
        score = scores[0]  # Extract the score from the list
        logger.info(f"Score: {score}")

        treshold = 0.3
        if score >= treshold:
            label = "<b>&#8252;&#65039; Spam DETECTED</b>"
            reason = f"score &#62;= {treshold}"
        else:
            label = "<i>No spam detected</i>"
            reason = f"score &#60; {treshold}"

        logger.info(f"The message was sent to the administrator and the group")
 
        spam_message_for_admins = (
            f"{(label)} <b>({score * 100}%)</b>\n"
            + "\n"
            + f"Канал: {(message.chat.title)}\n"
            + f"Автор: @{(message.from_user.username)}\n"
            + f"Время: {(message.date)}\n"
            + "\n"
            + f"{escape(text)}\n"
            + "\n"
            + f"Причина: {(reason)}\n"
        )

        spam_message_for_group = spam_message_for_admins
        for admin_id in ADMIN_IDS:
            await bot.send_message(admin_id,
                                spam_message_for_admins,
                                parse_mode="HTML")
        # Send the same message to the group
        if photo is None:
            await bot.send_message(GROUP_CHAT_ID,
                                spam_message_for_group,
                                arse_mode="HTML")
        else:
            photo_id = photo[-1].file_id
            await bot.send_photo(GROUP_CHAT_ID,
                                photo=photo_id,
                                caption=spam_message_for_group,
                                parse_mode="HTML")
