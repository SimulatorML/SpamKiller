import os
from dotenv import load_dotenv
from loguru import logger
from aiogram import Bot, Dispatcher, types
from add_new_user_id import load_new_members, add_new_member
from read_message import handle_msg_with_args
from functools import partial
from aiogram import executor

# Загрузить переменные среды из файла .env
load_dotenv()

logger.add(
    "logs/logs_from_bot.log", level="INFO"
)  # добавление логов где нужно сообщить, что все отработало корректно
logger.info("Инициализация бота")

TOKEN = os.getenv("API_KEY_TG")  # Получаем токен из переменной окружения
ADMIN_ID = os.getenv(
    "ADMIN_IDS"
)  # Получаем словарь ID администраторов из переменной окружения

bot = Bot(token=TOKEN)
dp = Dispatcher(
    bot
)  # Инициализируем диспетчера для бота, чтобы он мог обрабатывать сообщения

logger.info("Бот запущен")


async def on_startup(dp):
    load_new_members()
    await bot.send_message(ADMIN_ID, "Бот запущен")


logger.info("Бот остановлен")


async def on_shutdown(dp):
    await bot.send_message(ADMIN_ID, "Бот остановлен")
    await bot.close()


logger.info("Регистрация обработчика сообщений")


# Создаем обертку для handle_msg, передавая все необходимые аргументы
def handle_msg_partial():
    """
    Из-за того что в функции handle_msg_with_args есть дополнительные
    аргументы в виде bot и ADMIN_ID, то мы создаем обертку для для этой функции
    так как по умолчнию в обработчик сообщений передается только один аргумент

    Parameters
    ----------
    None

    Returns
    -------
    partial
        Обертка для handle_msg, передавая все необходимые аргументы
    """
    return partial(handle_msg_with_args, bot=bot, ADMIN_ID=ADMIN_ID)


# Регистрируем обработчик сообщений с переданными аргументами в фабрику декораторов
dp.message_handler()(handle_msg_partial())


# Обработка новых участников чата
@dp.message_handler(content_types=["new_chat_members"])
async def on_user_joined(message: types.Message):
    """
    Обработка новых участников чата

    Parameters
    ----------
    message : types.Message
        Сообщение с информацией о новых участниках чата

    Returns
    -------
    None
    """
    for user in message.new_chat_members:
        add_new_member(user)


if __name__ == "__main__":
    executor.start_polling(
        dp, on_startup=on_startup, on_shutdown=on_shutdown
    )  # Запускаем лонг поллинг
