import src.handlers
from aiogram import executor, types
from loguru import logger
from src.config import ADMIN_IDS
from src.app.loader import bot, dp
from src.utils.add_new_user_id import read_temp_list_with_new_user


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


if __name__ == '__main__':
    SpamKiller().start()
