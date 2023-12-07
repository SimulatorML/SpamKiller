from loguru import logger
from telethon import TelegramClient

from config import API_ID, API_HASH

CHANNEL = '@langchaindevchat'
OUTPUT_FILE = f'data/rules_base_model/whitelist_users_{CHANNEL}.txt'


async def scrape_messages(client: TelegramClient, channel: str = CHANNEL, output_file: str = OUTPUT_FILE):
    """Scraping messages from telegram channel"""
    # Connect to the client
    await client.start()
    logger.info("Client Created")
    logger.info("Scrapping...")

    user_message_count = {}
    whitelist_users = []

    # Accessing the channel
    async for message in client.iter_messages(CHANNEL, limit=10000):
        try:
            user_id = message.sender_id
            if user_id in user_message_count:
                user_message_count[user_id] += 1
            else:
                user_message_count[user_id] = 1
        except Exception:
            continue

    # Add users to whitelist if they sent 3 or more messages
    for user_id, messages_count in user_message_count.items():
        if messages_count >= 3:
            whitelist_users.append(user_id)

    with open(output_file, 'w') as file:
        for user_id in whitelist_users:
            file.write(str(user_id) + '\n')

    logger.info(f'Successfully scrapped messages and saved them in {output_file}')


if __name__ == '__main__':
    # Initialize the client
    client = TelegramClient('Scrape_user_ids', API_ID, API_HASH)

    # Running the client
    with client:
        client.loop.run_until_complete(scrape_messages(client=client))