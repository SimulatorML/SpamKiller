import os
import aiogram
from aiogram import executor, types
from loguru import logger
from dotenv import load_dotenv

# Local imports
from src.config import ADMIN_IDS
from src.app.loader import bot, dp
from src.utils.add_new_user_id import read_temp_list_with_new_user
from ban_user import BanManager  # Новый файл для управления спамом


# Инициализация менеджера спама
ban_manager = BanManager()

class SpamKiller:
    def start(self):
        executor.start_polling(
            dispatcher=dp, on_startup=self._on_startup, on_shutdown=self._on_shutdown
        )

    async def _on_startup(self, dp):
        logger.info("Bot started")
        read_temp_list_with_new_user()

        for admin_id in ADMIN_IDS:
            await bot.send_message(admin_id, "Bot started")

        await dp.bot.set_my_commands([
            types.BotCommand("add_id", "Add new admin"),
            types.BotCommand("del_id", "Delete admin"),
            types.BotCommand("update_whitelist", "Parse X messages in channel and add users with Y or more messages.\n"
                                                 "Example: channel X Y")
        ])

    async def _on_shutdown(self, dp):
        logger.info("Bot stopped")

        for admin_id in ADMIN_IDS:
            await bot.send_message(admin_id, "Bot stopped")
        await bot.close()


# Новый обработчик сообщений для фильтрации спама
@dp.message_handler()
async def handle_message(message: types.Message):
    user_id = str(message.from_user.id)
    classification = ban_manager.classify_message(message.text, user_id)

    if classification.startswith("Spam message deleted"):
        await message.delete()
        await message.reply("Your message was classified as spam.")
    elif classification.startswith("Definite spam"):
        await message.delete()
        await message.reply("You have been banned due to spam.")
    else:
        await message.reply("Message accepted.")


if __name__ == '__main__':
    SpamKiller().start()
