from aiogram import types
from src.app.loader import dp
from src.config import ADMIN_IDS
from src.utils.commands import add_admin, delete_admin, update_whitelist_users


@dp.message_handler(commands=["add_id"])
async def handle_add_admin(message: types.Message):
    global ADMIN_IDS
    await add_admin(message, ADMIN_IDS)


@dp.message_handler(commands=["del_id"])
async def handle_delete_admin(message: types.Message):
    global ADMIN_IDS
    await delete_admin(message, ADMIN_IDS)


@dp.message_handler(commands=["update_whitelist"])
async def handle_update_whitelist(message: types.Message):
    global ADMIN_IDS
    await update_whitelist_users(message, ADMIN_IDS)
