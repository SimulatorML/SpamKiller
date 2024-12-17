# SpamKiller


# What is the project about?
The problem of spam in large chats is common. Spam makes it difficult to communicate between people, search for the right information, which may eventually lead to people starting to leave the chat, because it will be impossible to be in it because of the abundance of spam.

By getting rid of spam in the chat, we will be able to make communication between people more comfortable, because it is unpleasant to correspond in the chat or search for information and often stumble upon fraudulent information about discounts of 90%, etc.

In general, the goal of this project is to reduce the amount of spam to a minimum, free administrators from routine viewing of the chat for spam, round-the-clock monitoring of the chat by a bot, timely decision-making on blocking the user and deleting spam.


# How does the Bot work?

### Overview
Our bot utilizes two main components to tackle spam: a primary AI model and a backup system. 

### Primary AI Model
At its core, the bot is powered by an AI model based on GPT-3.5-turbo. This model is adept at understanding and identifying various forms of spam, ensuring effective moderation.

### Backup System
In certain situations, such as when there are issues with the OpenAI API (like response delays or downtime), our bot switches to a secondary mechanism. This backup system operates on heuristic rules, ensuring that spam detection remains consistent even when the primary AI model is not available.

### Why This Approach?
This dual-system approach is designed to provide a seamless and reliable spam detection service. By utilizing the advanced capabilities of GPT-3.5-turbo, we ensure sophisticated spam filtering. Meanwhile, the heuristic rule-based backup maintains service stability, so our users always have a dependable line of defense against spam.

Our goal is to maintain a high-quality user experience, uninterrupted by spam and technical hiccups. This setup allows us to start delivering value immediately while continuously adapting and improving our service.

# Get Started
## Dependencies
* [DVC](https://dvc.org/doc/install)\
Install the required Python libraries and download the data:
```
make setup
make pull-data
```
## Environment Variables
* Specify the environment variables in `.env` file:
  - `API_KEY_SPAM_KILLER`: Bot Token
  - `TARGET_GROUP_ID`: Target group's ID
  - `TARGET_SPAM_ID`: Target spam group's ID
  - `TARGET_NOT_SPAM_ID`: Target not spam group's ID
  - `ADMIN_IDS`: User IDs of admins
  - `WHITELIST_ADMINS`: Whitelist of admins

## Bot Starting
* Start the bot by running bash-script:
```
make run
```

# Project structure
The following structure is used in this project:

1. `docs/`: contains project documentation
2. `data/`: not available on github (dvc pull to get access)

3. `logs/`: contains project logs,
    - `logs_from_bot.log` - the main log file in which all the actions of the bot are recorded (not available in the public version);
    - `temp_list_with_new_user.json` is a temporary file to which the user is added until he sends his first message to the chat

4. `static/`: contains script to launch the Bot
    - `run_bot.py ` - the main file in which the bot is launched

5. `scripts/`: contains scripts for working with data,
    - `data.py` - performs data cleaning
    - `make_metrics.py` - calculates the quality metrics of the model
    - `not_spam_id.py` - generate a CSV file with the not spam IDs
    - `predict_spam_scores.py` - makes predictions from the model
    - `watching.py` - in development

6. `src/`: contains the source code of the service
    - `app/`:
        * `bot.py`: source code for the Bot
        * `loader.py`: source code for initializating models, Bot, Dispatcher
    - `data/`:
        * `data.py` - source code for data cleaning
    - `handlers/`:
        * `commands.py`: handlers for adding and removing admin
        * `messages.py`: handler for processing new messages in chats
    - `metrics/`:
        * `metrics.py` - source code for calculating the quality metrics of the model
    - `models/`
        * `gpt_classifier.py`: source code for the GptSpamClassifier model used in production
        * `gpt_classifier_validation.py`: source code for the GptSpamClassifierValidation model for validation
        * `rules_base_model_prod.py` - source code for the RuleBasedClassifier model used in production
        * `rules_base_model_validation.py` - source code for the RuleBasedClassifier model for validation
    - `utils/`:
        * `add_new_user_id.py` - source code for adding a new user to the temporary file
        * `commands.py` - source code for bot commands
        * `message_processing.py` - source code for processing new messages
        * `spam_detection.py` - source code for spam detection process


# The main tools used in the project
![Pyhon](https://img.shields.io/badge/-Python_3.10.8-090909?style=for-the-badge&logo=python) ![Aiogram](https://img.shields.io/badge/-Aiogram_2.25.1-090909?style=for-the-badge&logo=Aiogram)    
![OpenAI](https://img.shields.io/badge/-openai_1.1.0-090909?style=for-the-badge&logo=openai&color=black)   
![Pandas](https://img.shields.io/badge/-pandas_1.3.0-090909?style=for-the-badge&logo=pandas) 
![Numpy](https://img.shields.io/badge/-Numpy_1.21.1-090909?style=for-the-badge&logo=Numpy) 
![Loguru](https://img.shields.io/badge/-Loguru_1.6.1-090909?style=for-the-badge&logo=xgboost) 
![Pylint](https://img.shields.io/badge/-Pylint_2.10.0-090909?style=for-the-badge&logo=Pylint)
