from aiogram import Bot, Dispatcher, types
from loguru import logger
from config import BOT_TOKEN
from src.models import RuleBasedClassifier, GptSpamClassifier
from src.utils.scrapper import Scrapper


# Add logger to file with level INFO
logger.add("logs/logs_from_spam_killer.log", level="INFO")

# Initialize models
gpt_classifier = GptSpamClassifier()
rule_based_classifier = RuleBasedClassifier()

# Initialize message scrapper
message_scrapper = Scrapper()

bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
logger.info("Initialized Bot")

# Initialize dispatcher for bot.
# Dispatcher is a class that process all incoming updates and handle them to registered handlers
dp = Dispatcher(bot)
