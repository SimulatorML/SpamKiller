import yaml
from aiogram import types
from typing import List
from loguru import logger
from src.app.loader import message_scrapper


def add_admin_id_to_env(new_id: str, lines: List[str]):
    for i, line in enumerate(lines):
        if line.startswith("ADMIN_IDS"):
            current_ids = line.strip().split("=")[1].split(",")
            if str(new_id) not in current_ids:
                current_ids.append(str(new_id))
                lines[i] = f"ADMIN_IDS={','.join(current_ids)}\n"
    return lines


def delete_admin_id_from_env(del_id: str, lines: List[str]):
    for i, line in enumerate(lines):
        if line.startswith("ADMIN_IDS"):
            current_ids = line.strip().split("=")[1].split(",")
            if str(del_id) in current_ids:
                current_ids.remove(str(del_id))
                lines[i] = f"ADMIN_IDS={','.join(current_ids)}\n"
    return lines


async def add_admin(message: types.Message, ADMIN_IDS: List[str]):
    # Check if message sender is an admin
    if str(message.from_user.id) not in ADMIN_IDS:
        await message.answer("Access denied")
        return

    new_id = message.get_args()
    if new_id and new_id.isdigit() and new_id not in ADMIN_IDS:
        ADMIN_IDS.append(new_id)
        with open(".env", "r") as f:
            lines = f.readlines()
        lines = add_admin_id_to_env(new_id, lines)
        with open(".env", "w") as f:
            f.writelines(lines)
        await message.answer("New admin added")
        logger.info("New moderator added")
    else:
        await message.answer("Invalid or duplicate ID")


async def delete_admin(message: types.Message, ADMIN_IDS: List[str]):
    # Check if message sender is an admin
    if str(message.from_user.id) not in ADMIN_IDS:
        await message.answer("Access denied")
        return

    del_id = message.get_args()
    if del_id and del_id.isdigit() and del_id in ADMIN_IDS:
        ADMIN_IDS.remove(del_id)
        with open(".env", "r") as f:
            lines = f.readlines()
        lines = delete_admin_id_from_env(del_id, lines)
        with open(".env", "w") as f:
            f.writelines(lines)
        await message.answer("Admin ID removed")
        logger.info("Moderator removed")
    else:
        await message.answer("Invalid ID or ID not found")

def add_user_to_whitelist(user_id: int):
    with open("./config.yml", "r") as f:
        config = yaml.safe_load(f)
        path_whitelist_users = config["whitelist_users"]

    with open(path_whitelist_users, 'a') as file:
        file.write('\n' + str(user_id))
    
    logger.info(f'User {user_id} was successfully added to whitelist_users.txt')

async def update_whitelist_users(message: types.Message, ADMIN_IDS):
    # Check if message sender is an admin
    if str(message.from_user.id) not in ADMIN_IDS:
        await message.answer("Access denied")
        return

    await message.answer('Scrapping...')
    channel = f'@{message.get_args().strip()}'
    parsed_ids = await message_scrapper.start(channel)
    parsed_ids = set(uid for uid in parsed_ids if uid is not None)

    # Read existing user IDs from the file
    try:
        with open("./config.yml", "r") as f:
            config = yaml.safe_load(f)
            path_whitelist_users = config["whitelist_users"]
        with open(path_whitelist_users, 'r') as file:
            existing_user_ids = set(file.read().splitlines())
            logger.info(len(existing_user_ids))
    except FileNotFoundError as e:
        logger.info(f'File {path_whitelist_users} not found')

    # Filter out user IDs that are already in the file
    new_user_ids = parsed_ids - existing_user_ids
    with open(path_whitelist_users, 'a') as file:
        for user_id in new_user_ids:
            file.write('\n' + str(user_id))

    logger.info(f'Whiltelist from {channel} updated, added {len(new_user_ids)} users')
    await message.answer(f'Whiltelist from {channel} updated, added {len(new_user_ids)} users')
