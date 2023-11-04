from aiogram import Bot, Dispatcher
from config import BOT_TOKEN
from src.models import RuleBasedClassifier


# Initialize classifier model
classifier = RuleBasedClassifier()

bot = Bot(token=BOT_TOKEN)
# Initialize dispatcher for bot. Dispatcher is a class that process all incoming updates and handle them to registered handlers
dp = Dispatcher(bot) 