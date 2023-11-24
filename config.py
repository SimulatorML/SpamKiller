import os
import yaml
import openai
import pandas as pd
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get OpenAI API Key from environment variable
PROXY_URL = os.getenv("PROXY_URL") # Get Proxy url if provided
THRESHOLD_RULE_BASED = 0.2 # Theshold for scores in rule based model. If score >= threshold, the message is considered as spam
BOT_TOKEN = os.getenv("API_KEY_SPAM_KILLER") # Get token from environment variable
ADMIN_IDS = (
    os.getenv("ADMIN_IDS").split(",") if os.getenv("ADMIN_IDS") else []
)  # Get admin id from environment variable (in .env file)
TARGET_GROUP_ID = (
    os.getenv("TARGET_GROUP_ID") if os.getenv("TARGET_GROUP_ID") else []
)  # Get tar  # Get target group id from environment variable (in .env file)
TARGET_SPAM_ID = (
    os.getenv("TARGET_SPAM_ID") if os.getenv("TARGET_SPAM_ID") else []
)  # Get target notid from environment variable (in .env file)
WHITELIST_ADMINS = (
    [int(i) for i in os.getenv("WHITELIST_ADMINS").split(",")]
    if os.getenv("WHITELIST_ADMINS")
    else []
) # Get whitelist admins
TARGET_NOT_SPAM_ID = (
    os.getenv("TARGET_NOT_SPAM_ID") if os.getenv("TARGET_NOT_SPAM_ID") else []
)  # Get target not spam group id from environment variable (in .env file)
AUTHORIZED_USER_IDS = (
    os.getenv("AUTHORIZED_USER_IDS").split(",")
    if os.getenv("AUTHORIZED_USER_IDS")
    else []
)
AUTHORIZED_GROUP_IDS = (
    os.getenv("AUTHORIZED_GROUP_IDS").split(",")
    if os.getenv("AUTHORIZED_GROUP_IDS")
    else []
)  # Get admin id from environment variable (in .env file)

# Get whitelist users
with open("./config.yml", "r") as f:
    config = yaml.safe_load(f)
    path_whitelist_users = config["whitelist_users"]

with open(path_whitelist_users, 'r') as file:
    WHITELIST_USERS = [int(user_id.strip()) for user_id in file.readlines()]

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.2,
    "max_tokens": 40,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
} # Parameters for GPT