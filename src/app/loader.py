from aiogram import Bot, Dispatcher
from loguru import logger
from config import BOT_TOKEN
from src.models import RuleBasedClassifier


# Add logger to file with level INFO
logger.add("logs/logs_from_spam_killer.log", level="INFO")

# Initialize classifier model
classifier = RuleBasedClassifier()
logger.info("Initialized Model")

bot = Bot(token=BOT_TOKEN)
logger.info("Initialized Bot")

# Initialize dispatcher for bot. Dispatcher is a class that process all incoming updates and handle them to registered handlers
dp = Dispatcher(bot) 