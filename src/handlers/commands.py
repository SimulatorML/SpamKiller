from aiogram import types
from loguru import logger
from aiogram.dispatcher import FSMContext
from src.app.loader import dp
from src.config import ADMIN_IDS
from src.utils.commands import (
    add_admin, 
    delete_admin, 
    update_whitelist_users, 
    add_to_goldlist,
    delete_from_goldlist
)

# State identifiers
ADD_ID_STATE = "add_id_state"
DEL_ID_STATE = "del_id_state"
UPDATE_STATE = "update_state"
ADD_GOLDLIST_STATE = "add_goldlist_state"
DEL_GOLDLIST_STATE = "del_goldlist_state"

@dp.message_handler(commands=["add_id"])
async def handle_add_admin(message: types.Message, state: FSMContext):
    await message.answer("Please enter id")
    await state.set_state(ADD_ID_STATE)


@dp.message_handler(commands=["del_id"])
async def handle_delete_admin(message: types.Message, state: FSMContext):
    await state.set_state(DEL_ID_STATE)
    await message.answer("Please enter id")


@dp.message_handler(commands=["update_whitelist"])
async def handle_update_whitelist(message: types.Message, state: FSMContext):
    await state.set_state(UPDATE_STATE)
    await message.answer("Please enter the channel name and message count (optional)")


@dp.message_handler(commands=["add_goldlist"])
async def handle_add_goldlist(message: types.Message, state: FSMContext):
    await state.set_state(ADD_GOLDLIST_STATE)
    await message.answer("Пожалуйста, введите ID пользователя для добавления в goldlist")


@dp.message_handler(commands=["del_goldlist"])
async def handle_delete_goldlist(message: types.Message, state: FSMContext):
    await state.set_state(DEL_GOLDLIST_STATE)
    await message.answer("Пожалуйста, введите ID пользователя для удаления из goldlist")


# Special handler to process input from user
@dp.message_handler(state=[ADD_ID_STATE, DEL_ID_STATE, UPDATE_STATE, ADD_GOLDLIST_STATE, DEL_GOLDLIST_STATE])
async def process_argument(message: types.Message, state: FSMContext):
    global ADMIN_IDS
    current_state = await state.get_state()
    if current_state == ADD_ID_STATE:
        await add_admin(message, ADMIN_IDS)
    elif current_state == DEL_ID_STATE:
        await delete_admin(message, ADMIN_IDS)
    elif current_state == UPDATE_STATE:
        await update_whitelist_users(message, ADMIN_IDS)
    elif current_state == ADD_GOLDLIST_STATE:
        await add_to_goldlist(message, ADMIN_IDS)
    elif current_state == DEL_GOLDLIST_STATE:
        await delete_from_goldlist(message, ADMIN_IDS)
    await state.finish()  # Reset the state


@dp.message_handler(state=ADD_GOLDLIST_STATE)
async def process_argument(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    
    try:
        # Получаем список администраторов чата
        channel_admins_info = await message.bot.get_chat_administrators(message.chat.id)
        admins = [str(admin.user.id) for admin in channel_admins_info]
        
        # Проверяем, является ли отправитель администратором
        if str(message.from_user.id) not in ADMIN_IDS and str(message.from_user.id) not in admins:
            await message.answer("Доступ запрещен")
            await state.finish()
            return

        if current_state == ADD_GOLDLIST_STATE:
            await add_to_goldlist(message, ADMIN_IDS)
        elif current_state == DEL_GOLDLIST_STATE:
            await delete_from_goldlist(message, ADMIN_IDS)
        await state.finish()
        
    except Exception as e:
        logger.error(f"Ошибка при проверке прав администратора: {e}")
        await message.answer("Произошла ошибка при проверке прав доступа")
        await state.finish()


@dp.message_handler(state=DEL_GOLDLIST_STATE)
async def process_delete_goldlist(message: types.Message, state: FSMContext):
    await delete_from_goldlist(message, ADMIN_IDS)
    await state.finish()

