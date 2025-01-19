import sys
import yaml
from aiogram import types
from typing import List
from loguru import logger
from src.app.loader import message_scrapper
import os


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

    with open(path_whitelist_users, "a") as file:
        file.write("\n" + str(user_id))

    logger.info(f"User {user_id} was successfully added to whitelist_users.txt")


async def update_whitelist_users(message: types.Message, ADMIN_IDS):
    sys.tracebacklimit = 0
    # Check if message sender is an admin
    if str(message.from_user.id) not in ADMIN_IDS:
        await message.answer("Access denied")
        return

    command = message.text.split(" ")
    args = []
    channel = f"@{command[0]}"
    args.append(channel)
    await message.answer("Updating...")
    if len(command) > 1:
        try:
            try:
                depth = int(command[1])
                threshold = int(command[2])
            except ValueError:
                # Handle errors specifically during integer conversion
                logger.exception("Error converting command arguments to integer")
                await message.answer("Number of messages must be integer")
                return
            if depth <= 0 or threshold <= 0:
                logger.exception("Error converting command arguments to integer")
                # await message.answer('Number of messages must be positive')
                raise ValueError("Number of messages must be positive")
            args.append(depth)
            args.append(threshold)
        except ValueError as ve:
            logger.exception(f"Error parsing command : {ve}")
            await message.answer(f"{ve}")
            return
        except Exception as ex:
            logger.exception(f"Error parsing command: {ex}")
            await message.answer("Error...Check command and try again")
            return

    parsed_ids = await message_scrapper.start(args)
    if parsed_ids:
        parsed_ids = set(uid for uid in parsed_ids if uid is not None)
        # Read existing user IDs from the file
        try:
            with open("./config.yml", "r") as f:
                config = yaml.safe_load(f)
                path_whitelist_users = config["whitelist_users"]
            with open(path_whitelist_users, "r+") as file:
                existing_user_ids = set(file.read().splitlines())
                # Filter out user IDs that are already in the file
                new_user_ids = parsed_ids - existing_user_ids
                file.seek(0, 2)  # go to the end of the file
                for user_id in new_user_ids:
                    file.write("\n" + str(user_id))

            result_message = (
                f"Whitelist from {channel} updated, added {len(new_user_ids)} user(s), "
                f"total {len(existing_user_ids) + len(new_user_ids)}"
            )
        except FileNotFoundError:
            logger.info(f"File {path_whitelist_users} not found")
            result_message = f"File {path_whitelist_users} not found"

    else:
        result_message = "Failed updating whitelist. See logs for more information"
    logger.info(result_message)
    await message.answer(result_message)


async def add_to_goldlist(message: types.Message, ADMIN_IDS: List[str]):
    # Получаем список администраторов чата
    try:
        channel_admins_info = await message.bot.get_chat_administrators(message.chat.id)
        admins = [str(admin.user.id) for admin in channel_admins_info]
        
        # Проверяем, является ли отправитель администратором
        if str(message.from_user.id) not in ADMIN_IDS and str(message.from_user.id) not in admins:
            await message.answer("Доступ запрещен")
            return

        user_id = message.text  # Получаем ID из текста сообщения
        if not user_id or not user_id.isdigit():
            await message.answer("Некорректный ID пользователя")
            return

        # Загружаем конфигурацию
        with open("./config.yml", "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'goldlist_users' not in config:
            logger.error("В config.yml отсутствует параметр goldlist_users")
            await message.answer("Ошибка конфигурации")
            return
            
        path_goldlist_users = config["goldlist_users"]
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path_goldlist_users), exist_ok=True)

        # Читаем существующие ID
        existing_ids = set()
        if os.path.exists(path_goldlist_users):
            with open(path_goldlist_users, "r", encoding='utf-8') as f:
                existing_ids = set(line.strip() for line in f.readlines() if line.strip())

        # Проверяем, не существует ли уже такой ID
        if user_id in existing_ids:
            await message.answer("Пользователь уже в goldlist")
            return

        # Добавляем новый ID
        with open(path_goldlist_users, "a", encoding='utf-8') as f:
            if existing_ids:
                f.write(f"\n{user_id}")
            else:
                f.write(user_id)

        await message.answer(f"Пользователь {user_id} успешно добавлен в goldlist")
        logger.info(f"Пользователь {user_id} добавлен в goldlist")

    except yaml.YAMLError as e:
        logger.error(f"Ошибка при чтении config.yml: {e}")
        await message.answer("Ошибка при чтении конфигурации")
    except Exception as e:
        logger.error(f"Ошибка при добавлении в goldlist: {e}")
        await message.answer("Произошла ошибка при добавлении пользователя")


async def delete_from_goldlist(message: types.Message, ADMIN_IDS: List[str]):
    # Проверяем права администратора
    try:
        channel_admins_info = await message.bot.get_chat_administrators(message.chat.id)
        admins = [str(admin.user.id) for admin in channel_admins_info]
        
        if str(message.from_user.id) not in ADMIN_IDS and str(message.from_user.id) not in admins:
            await message.answer("Доступ запрещен")
            return

        # Получаем ID из текста сообщения напрямую
        user_id = message.text
        if not user_id or not user_id.isdigit():
            await message.answer("Некорректный ID пользователя")
            return

        # Загружаем конфигурацию
        with open("./config.yml", "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if 'goldlist_users' not in config:
            logger.error("В config.yml отсутствует параметр goldlist_users")
            await message.answer("Ошибка конфигурации")
            return
            
        path_goldlist_users = config["goldlist_users"]

        # Проверяем существование файла
        if not os.path.exists(path_goldlist_users):
            await message.answer("Файл goldlist не найден")
            return

        # Читаем существующие ID
        with open(path_goldlist_users, "r", encoding='utf-8') as f:
            existing_ids = [line.strip() for line in f.readlines() if line.strip()]

        # Проверяем, существует ли ID в списке
        if user_id not in existing_ids:
            await message.answer("Пользователь не найден в goldlist")
            return

        # Удаляем ID из списка
        existing_ids.remove(user_id)
        
        # Записываем обновленный список
        with open(path_goldlist_users, "w", encoding='utf-8') as f:
            if existing_ids:  # Проверяем, не пустой ли список
                f.write("\n".join(existing_ids))
            else:
                f.write("")  # Записываем пустую строку, если список пуст

        await message.answer(f"Пользователь {user_id} успешно удален из goldlist")
        logger.info(f"Пользователь {user_id} удален из goldlist")

    except yaml.YAMLError as e:
        logger.error(f"Ошибка при чтении config.yml: {e}")
        await message.answer("Ошибка при чтении конфигурации")
    except Exception as e:
        logger.error(f"Ошибка при удалении из goldlist: {e}")
        await message.answer("Произошла ошибка при удалении пользователя")
