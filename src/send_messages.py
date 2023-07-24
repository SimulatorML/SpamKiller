from loguru import logger
from html import escape


# Reading only new messages from new users
async def handle_msg_with_args(
    message,
    bot,
    classifier,
    ADMIN_IDS,
    GROUP_CHAT_ID,
    AUTHORIZED_USER_IDS,
    AUTHORIZED_GROUP_IDS,
    TARGET_SPAM_ID,
    TARGET_NOT_SPAM_ID,
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

    # await check_user_id(message):
    # if (
    #     str(message.chat.id) in AUTHORIZED_GROUP_IDS
    #     or str(message.from_id) in AUTHORIZED_USER_IDS
    # ):
    logger.info(f"Message got from new user. Checking for spam")

    reply_to_message_id = (
        message.reply_to_message.message_id if message.reply_to_message else None
    )
    photo = message.photo[-1].file_id if message.photo else None
    text = message.text or message.caption or ""
    print(message)
    print(message.chat.id)
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
    score, features = classifier.predict(X)
    logger.info(f"Score: {score}")

    treshold = 0.3
    if score >= treshold:
        label = "<b>&#8252;&#65039; Spam DETECTED</b>"
    else:
        label = "<i>No spam detected</i>"
    if len(text) > 600:
        text = text[:600] + '...'
    logger.info("The message was sent to the administrator and the group")
    if len(features.split('-')) > 2:
        spam_message_for_admins = (
            f"{(label)} <b>({round(score * 100, 2)}%)</b>\n"
            + "\n"
            + f"Канал: {(message.chat.title)}\n"
            + f"Автор: @{(message.from_user.username)}\n"
            + f"Время: {(message.date)}\n"
            + "\n"
            + f"{escape(text)}\n"
            + "\n"
            + "Причины:\n"
            + features
        )
    else:
        spam_message_for_admins = (
            f"{(label)} <b>({round(score * 100, 2)}%)</b>\n"
            + "\n"
            + f"Канал: {(message.chat.title)}\n"
            + f"Автор: @{(message.from_user.username)}\n"
            + f"Время: {(message.date)}\n"
            + "\n"
            + f"{escape(text)}\n"
            + "\n"
            + "Причина:\n"
            + features
        )

    spam_message_for_group = spam_message_for_admins
    # Send the same message to the groupы
    if photo is None:
        if score >= 0.1:
            await bot.send_message(GROUP_CHAT_ID, spam_message_for_group, parse_mode="HTML")
        for admin_id in ADMIN_IDS:
            await bot.send_message(admin_id, spam_message_for_admins, parse_mode="HTML")

        if score >= treshold:
            await bot.send_message(
                TARGET_SPAM_ID, spam_message_for_admins, parse_mode="HTML"
            )
            await bot.delete_message(message.chat.id, message.message_id)
        else:
            await bot.send_message(
                TARGET_NOT_SPAM_ID, spam_message_for_admins, parse_mode="HTML"
            )

    else:
        if score >= 0.1:
            await bot.send_photo(
                GROUP_CHAT_ID,
                photo=photo,
                caption=spam_message_for_admins,
                parse_mode="HTML",
            )
        for admin_id in ADMIN_IDS:
            await bot.send_photo(
                admin_id,
                photo=photo,
                caption=spam_message_for_admins,
                parse_mode="HTML",
            )

        if score >= treshold:
            await bot.send_photo(
                TARGET_SPAM_ID,
                photo=photo,
                caption=spam_message_for_admins,
                parse_mode="HTML",
            )
            await bot.delete_message(message.chat.id, message.message_id)
        else:
            await bot.send_photo(
                TARGET_NOT_SPAM_ID,
                photo=photo,
                caption=spam_message_for_admins,
                parse_mode="HTML",
            )
