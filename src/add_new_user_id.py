import json
from aiogram import types
from loguru import logger
from datetime import datetime

temp_list_with_new_user = []


def save_new_members():  # Saving new users
    logger.info("Save new users")  # Save new users
    with open(
        "logs/temp_list_with_new_user.json", "w"
    ) as file:  # Opening a file for writing
        json.dump(temp_list_with_new_user, file)  # Writing data to a file


def read_temp_list_with_new_user():  # ead new users added berfore
    logger.info("Load new users")  # Load new users
    global temp_list_with_new_user  # Global variable
    try:  # if file not found
        with open(
            "logs/temp_list_with_new_user.json", "r"
        ) as file:  # Opening a file for reading
            temp_list_with_new_user = json.load(file)  # Reading data from a file
    except FileNotFoundError:  # if file not found
        pass


def add_new_member(user):  # Adding a new user from def on_user_joined from app.py
    logger.info("Adding a new user in temp list ")  # Adding a new user
    global temp_list_with_new_user  # Global variable
    # load current list of new members from the file
    read_temp_list_with_new_user()
    user_info = {
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "join_time": datetime.now().isoformat(),
    }  # Information about the user
    temp_list_with_new_user.append(user_info)  # Adding a user to the list
    save_new_members()  # Saving the list


async def check_user_id(message: types.Message):  # Checking the user ID
    global temp_list_with_new_user  # Global variable
    user_id = message.from_user.id  # User ID
    for member in temp_list_with_new_user:  # Going through the list
        if member["id"] == user_id:  # If the user is in the list
            temp_list_with_new_user.remove(member)  # Removing a user from the list
            save_new_members()  # Saving the list
            return True
    return False
