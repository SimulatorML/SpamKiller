from aiogram import types
from aiogram.dispatcher import FSMContext
from src.app.loader import dp
from src.config import ADMIN_IDS
from src.utils.commands import add_admin, delete_admin, update_whitelist_users

# State identifiers
ADD_ID_STATE = "add_id_state"
DEL_ID_STATE = "del_id_state"
UPDATE_STATE = "update_state"

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


# Special handler to process input from user
@dp.message_handler(state=[ADD_ID_STATE, DEL_ID_STATE, UPDATE_STATE])
async def process_argument(message: types.Message, state: FSMContext):
    global ADMIN_IDS
    current_state = await state.get_state()
    if current_state == ADD_ID_STATE:
        await add_admin(message, ADMIN_IDS)
    elif current_state == DEL_ID_STATE:
        await delete_admin(message, ADMIN_IDS)
    elif current_state == UPDATE_STATE:
        await update_whitelist_users(message, ADMIN_IDS)
    await state.finish()  # Reset the state

