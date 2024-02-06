from aiogram import types
from functools import partial
from loguru import logger
from config import (
    THRESHOLD_RULE_BASED,
    ADMIN_IDS,
    TARGET_GROUP_ID,
    AUTHORIZED_USER_IDS,
    AUTHORIZED_GROUP_IDS,
    TARGET_SPAM_ID,
    TARGET_NOT_SPAM_ID,
    WHITELIST_ADMINS,
)
from src.app.loader import bot, dp, gpt_classifier, rule_based_classifier
from src.utils.spam_detection import handle_msg_with_args
from src.utils.add_new_user_id import add_new_member


def handle_msg_partial():
    """
    Due to the fact that the handle_msg_with_args function has additional
    arguments in the form of bot and ADMIN_ID, then we create a wrapper for this function
    since by default only one argument is passed to the message handler

    Parameters
    ----------
    None

    Returns
    -------
    partial
        Wrapper for handle_msg, passing all the necessary arguments
    """
    return partial(
        handle_msg_with_args,
        bot=bot,
        gpt_classifier=gpt_classifier,
        rule_based_classifier=rule_based_classifier,
        THRESHOLD_RULE_BASED=THRESHOLD_RULE_BASED,
        ADMIN_IDS=ADMIN_IDS,
        GROUP_CHAT_ID=TARGET_GROUP_ID,
        AUTHORIZED_USER_IDS=AUTHORIZED_USER_IDS,
        AUTHORIZED_GROUP_IDS=AUTHORIZED_GROUP_IDS,
        TARGET_SPAM_ID=TARGET_SPAM_ID,
        TARGET_NOT_SPAM_ID=TARGET_NOT_SPAM_ID,
        WHITELIST_ADMINS=WHITELIST_ADMINS,
    )  # Creating a wrapper for handle_msg, passing all the necessary arguments


# Registering a message handler with the arguments passed to the decorator factory
logger.info("Register handlers")
dp.message_handler(
    content_types=["any"],
)(handle_msg_partial())  # Registering a message handler


# Processing new chat users
@dp.message_handler(
    content_types=["new_chat_members"]
)  # Decorator for processing new chat users
async def on_user_joined(
    message: types.Message,
):  # Message handler for processing new chat users
    """
    Processing new chat users

    Parameters
    ----------
    message : types.Message
        Message from new user

    Returns
    -------
    None
    """
    for user in message.new_chat_members:  # Iterating over new chat users
        add_new_member(user)  # Adding new chat user to database
