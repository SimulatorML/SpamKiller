# SpamKiller


# What is the project about?
The problem of spam in large chats is common. Spam makes it difficult to communicate between people, search for the right information, which may eventually lead to people starting to leave the chat, because it will be impossible to be in it because of the abundance of spam.

By getting rid of spam in the chat, we will be able to make communication between people more comfortable, because it is unpleasant to correspond in the chat or search for information and often stumble upon fraudulent information about discounts of 90%, etc.

In general, the goal of this project is to reduce the amount of spam to a minimum, free administrators from routine viewing of the chat for spam, round-the-clock monitoring of the chat by a bot, timely decision-making on blocking the user and deleting spam.


# How does the bot work?
This version of the bot uses a primitive model based on heuristic rules. This is justified in order to start with a simple one and complicate the model step by step and understand whether it is worth spending human resources on developing a more complex model when a simpler model can cope with the task.


# Project structure
The following structure is used in this project:
1. `src module`: contains the main code of the project,
    - `app.py ` - the main file in which the bot is launched

2. `data module`: not available in the public version

3. `logs module`: contains project logs,
    - `logs_from_bot.log` - the main log file in which all the actions of the bot are recorded (not available in the public version);
    - `temp_list_with_new_user.json` is a temporary file to which the user is added until he sends his first message to the chat

4. `scripts modele`: contains scripts for working with data,
    - `data_preprocessing.py` - performs data cleaning
    - `make_metrics.py` - calculates the quality metrics of the model
    - `predict_spam_scores.py` - makes predictions from the model
    - `watching.py` - in development
5. `src module`: contains the source code
    - `data module` `data.py` - source code for data cleaning
    - `model module`
    - `model.py` - source code for the model
    - `scripts module`
    - `make_metrics.py` - source code for calculating the quality metrics of the model
    - `add_new_user_id.py` - source code for adding a new user to the temporary file
    - `commands.py` - source code for bot commands
    - `send_messages.py` - source code for sending messages to admins and a group







The following structure is used in this project:
1. `src module`: contains the main code of the project,
    1.1 `app.py ` - the main file in which the bot is launched

2. `data module`: not available in the public version

3. `logs module`: contains project logs,
    3.1 `logs_from_bot.log` - the main log file in which all the actions of the bot are recorded (not available in the public version);
    3.2 `temp_list_with_new_user.json` is a temporary file to which the user is added until he sends his first message to the chat

4. `scripts modele`: contains scripts for working with data,
    4.1 `data_preprocessing.py` - performs data cleaning
    4.2 `make_metrics.py` - calculates the quality metrics of the model
    4.3 `predict_spam_scores.py` - makes predictions from the model
    4.4 `watching.py` - in development
5. `src module`: contains the source code
    5.1 data module
        5.1.1 `data.py` - source code for data cleaning
    5.2 `model module`
        5.2.1 `model.py` - source code for the model
    5.3 scripts module
        5.3.1 `make_metrics.py` - source code for calculating the quality metrics of the model
    5.4 add_new_user_id.py - source code for adding a new user to the temporary file
    5.5 `commands.py` - source code for bot commands
    5.6  `send_messages.py` - source code for sending messages to admins and a group








# The main tools used in the project
![Pyhon](https://img.shields.io/badge/-Python_3.8.15-090909?style=for-the-badge&logo=python) ![Aiogram](https://img.shields.io/badge/-Aiogram_2.25.1-090909?style=for-the-badge&logo=Aiogram)       ![Pandas](https://img.shields.io/badge/-pandas_1.3.0-090909?style=for-the-badge&logo=pandas) ![Numpy](https://img.shields.io/badge/-Numpy_1.21.1-090909?style=for-the-badge&logo=Numpy) ![Loguru](https://img.shields.io/badge/-Loguru_1.6.1-090909?style=for-the-badge&logo=xgboost) ![Pylint](https://img.shields.io/badge/-Pylint_2.10.0-090909?style=for-the-badge&logo=Pylint)
