from aiogram import types
from typing import List
from loguru import logger


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
