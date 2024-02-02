from loguru import logger
from telethon import TelegramClient
from config import API_ID, API_HASH

class Scrapper:
    """A class for scraping Telegram messages from specified channels using the Telethon library."""
    def __init__(self, api_id: int = API_ID, api_hash: str = API_HASH, N: int = 10000, n: int = 3):
        self.api_id = api_id
        self.api_hash = api_hash
        self.N = N
        self.n = n
        self.client = TelegramClient('Scrape_user_ids', self.api_id, self.api_hash)

        logger.info("Initialized Scrapper")

    async def _scrape_messages(self, channel: str):
        """Scraping messages from telegram channel"""
        # Connect to the client
        await self.client.start()
        logger.info("Client Created")
        logger.info("Scrapping...")
        user_message_count = {}
        whitelist_users = []
        # Accessing the channel
        async for message in self.client.iter_messages(channel, limit=self.N):
            try:
                user_id = str(message.sender_id)
                if user_id in user_message_count:
                    user_message_count[user_id] += 1
                else:
                    user_message_count[user_id] = 1
            except Exception as e:
                logger.warning(f'Error scrapping: {e}')
                continue

        # Add users to whitelist if they sent n or more messages
        for user_id, messages_count in user_message_count.items():
            if messages_count >= self.n:
                whitelist_users.append(user_id)

        logger.info('Successfully scrapped messages')
        return whitelist_users

    async def start(self, channel):
        async with self.client:
            return await self._scrape_messages(channel)
