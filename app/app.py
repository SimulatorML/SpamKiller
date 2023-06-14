import os
from dotenv import load_dotenv
from loguru import logger
from aiogram import Bot, Dispatcher, types
from src.add_new_user_id import read_temp_list_with_new_user, add_new_member, check_user_id
from functools import partial
from aiogram import executor
from src.models.rules_base_model import RuleBasedClassifier  # Импортируем наш класс


# Load environment variables
load_dotenv()

logger.add("logs/logs_from_bot.log", level="INFO")  # Add logger to file with level INFO
logger.info("Init bot")

TOKEN = os.getenv("API_KEY_TG")  # Get token from environment variable
ADMIN_ID = os.getenv(
    "ADMIN_IDS"
)  # Get admin id from environment variable (in .env file)

classifier = RuleBasedClassifier()  # Создаем экземпляр нашего классификатора

bot = Bot(token=TOKEN)
dp = Dispatcher(
    bot
)  # Initialize dispatcher for bot. Dispatcher is a class that process all incoming updates and handle them to registered handlers


async def on_startup(dp):
    read_temp_list_with_new_user()
    await bot.send_message(ADMIN_ID, "Bot started")


logger.info("Bot stopped")


async def on_shutdown(dp):
    logger.info("Bot started")
    await bot.send_message(ADMIN_ID, "Bot stopped")
    await bot.close()

async def handle_msg_with_args(message: types.Message, bot, ADMIN_ID):
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
    if await check_user_id(message):
        logger.info(f"Message got from new user. Checking for spam")

        score = classifier.score(message.text)
        logger.info(f"Score: {score}")

        if score >= 0.90:
            logger.info(f"The message suspected of spam was sent to the administrator")
            await bot.send_message(
                ADMIN_ID,
                f'A message suspected of spam with a probability of {int(score * 100)} percent -->"{message.text}" <-- from user {message.from_user.id}',
            )

# Creating a wrapper for handle_msg, passing all the necessary arguments
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
    return partial(handle_msg_with_args, bot=bot, ADMIN_ID=ADMIN_ID) # Creating a wrapper for handle_msg


# Registering a message handler with the arguments passed to the decorator factory
logger.info("Register handlers")
dp.message_handler()(handle_msg_partial()) # Registering a message handler


# Processing new chat users
@dp.message_handler(content_types=["new_chat_members"]) # Decorator for processing new chat users
async def on_user_joined(message: types.Message): # Message handler for processing new chat users
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
    for user in message.new_chat_members: # Iterating over new chat users
        add_new_member(user) # Adding new chat user to database


if __name__ == "__main__":
    executor.start_polling(
        dp, on_startup=on_startup, on_shutdown=on_shutdown
    )  # Launching long polling
