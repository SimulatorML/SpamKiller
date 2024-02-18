from loguru import logger
from src.config import WHITELIST_USERS
from src.utils.commands import add_user_to_whitelist
from src.utils.message_processing import (
    extract_entities,
    build_data_frame,
    classify_message,
    send_spam_alert,
)


# Reading only new messages from new users
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
    """
    logger.info(
        f"Got new message from a user {message.from_id} in {message.chat.id} ({message.chat.username}). Checking for spam..."
    )

    # Getting features
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

    logger.debug(f"Chat_id: {message.chat.id}")
    logger.debug(f"User_id: {message.from_id}")
    logger.debug(f"User_bio: {user_description}")
    logger.debug(f"Message_text: {text}")
    logger.debug(f"Spoiler_link: {spoiler_link}")
    logger.debug(f"hidden_link: {hidden_link}")
    logger.debug(f"Contains photo: {True if photo else False}")
    logger.debug(f"Channel: {channel}")

    # Building DataFrame
    X = build_data_frame(
        text=text,
        bio=user_description,
        from_id=message.from_id,
        photo=photo,
        reply_to_message_id=reply_to_message_id,
        channel=channel,
    )
    print(X)

    # Getting administrators of the channel
    channel_admins_info = await bot.get_chat_administrators(message.chat.id)
    admins = [admin.user.id for admin in channel_admins_info]

    # Classifying message
    msg_features = await classify_message(
        X=X,
        gpt_classifier=gpt_classifier,
        rule_based_classifier=rule_based_classifier,
        THRESHOLD_RULE_BASED=THRESHOLD_RULE_BASED,
        admins=admins,
        WHITELIST_ADMINS=WHITELIST_ADMINS,
        WHITELIST_USERS=WHITELIST_USERS,
    )

    if (
        (msg_features["label"] == 0)
        and (message.from_id not in WHITELIST_USERS)
        and msg_features["model_name"] in ["GptSpamClassifier", "RuleBasedClassifier"]
    ):
        # If the message is predicted as not-spam and the user is not in WHITELIST_USERS,
        # user_id will be added to whitelist users
        WHITELIST_USERS.append(message.from_id)
        add_user_to_whitelist(user_id=message.from_id)

    logger.info(f"Label: {'Spam' if msg_features['label'] == 1 else 'Not-Spam'}")

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
