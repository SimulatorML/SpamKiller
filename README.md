# SpamKiller


# What is the project about?
The problem of spam in large chats is common. Spam makes it difficult to communicate between people, search for the right information, which may eventually lead to people starting to leave the chat, because it will be impossible to be in it because of the abundance of spam.

By getting rid of spam in the chat, we will be able to make communication between people more comfortable, because it is unpleasant to correspond in the chat or search for information and often stumble upon fraudulent information about discounts of 90%, etc.

In general, the goal of this project is to reduce the amount of spam to a minimum, free administrators from routine viewing of the chat for spam, round-the-clock monitoring of the chat by a bot, timely decision-making on blocking the user and deleting spam.


# How does the bot work?
This version of the bot uses a primitive model based on heuristic rules. This is justified in order to start with a simple one and complicate the model step by step and understand whether it is worth spending human resources on developing a more complex model when a simpler model can cope with the task.


# Project structure
The following structure is used in this project
: the `src` module: contains the main code of the project, namely:
`app.py ` - the main file in which the bot is launched; `add_new_user_id.py ` - adding a new user to a temporary file.json, the user will stay in it until he sends his first message to the chat; `read_message.py ` - reading the very first message left by the newly added user and checking it for spam, with a further message to the chat administrator if the message is recognized as spam; `json_to_csv.py ` is the file responsible for extracting text from .a json file containing the exported chat history with and without spam, followed by translation into a .csv table; `clean_text.py ` - in this file, the text is cleared for measuring the quality of the model and providing spam samples for detection by the bot.


# The main tools used in the project
![Pyhon](https://img.shields.io/badge/-Python_3.8.15-090909?style=for-the-badge&logo=python) ![Aiogram](https://img.shields.io/badge/-Aiogram_2.25.1-090909?style=for-the-badge&logo=Aiogram)       ![Pandas](https://img.shields.io/badge/-pandas_1.3.0-090909?style=for-the-badge&logo=pandas) ![Numpy](https://img.shields.io/badge/-Numpy_1.21.1-090909?style=for-the-badge&logo=Numpy) ![Loguru](https://img.shields.io/badge/-Loguru_1.6.1-090909?style=for-the-badge&logo=xgboost) ![Pylint](https://img.shields.io/badge/-Pylint_2.10.0-090909?style=for-the-badge&logo=Pylint)
