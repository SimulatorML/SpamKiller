import json
from aiogram import types
from loguru import logger
from datetime import datetime

temp_list_with_new_user = []

logger.info("Загрузка новых пользователей")  # загрузка новых пользователей


def save_new_members():  # сохранение новых пользователей
    with open(
        "logs/temp_list_with_new_user.json", "w"
    ) as file:  # открываем файл на запись
        json.dump(temp_list_with_new_user, file)  # записываем в файл


logger.info("Сохранение новых пользователей")


def load_new_members():  # загрузка новых пользователей
    global temp_list_with_new_user  # глобальная переменная
    try:  # попытка открыть файл
        with open(
            "logs/temp_list_with_new_user.json", "r"
        ) as file:  # открываем файл на чтение
            temp_list_with_new_user = json.load(file)  # загружаем данные из файла
    except FileNotFoundError:  # если файл не найден
        pass


def add_new_member(user):  # добавление нового пользователя
    global temp_list_with_new_user  # глобальная переменная
    user_info = {
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "join_time": datetime.now().isoformat(),
    }  # информация о пользователе
    temp_list_with_new_user.append(user_info)  # добавляем пользователя в список
    save_new_members()  # сохраняем список


logger.info("Проверка ID пользователя")  # проверка ID пользователя


async def check_user_id(message: types.Message):  # проверка ID пользователя
    global temp_list_with_new_user  # глобальная переменная
    user_id = message.from_user.id  # ID пользователя
    for member in temp_list_with_new_user:  # перебираем список пользователей
        if member["id"] == user_id:  # если ID пользователя есть в списке
            temp_list_with_new_user.remove(member)  # удаляем пользователя из списка
            save_new_members()  # сохраняем список
            return True
    return False
