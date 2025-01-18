import pandas as pd
import asyncio
import yaml
import re
from dataclasses import dataclass
from fuzzywuzzy import fuzz
from typing import Dict, List
from telethon.sync import TelegramClient
from telethon.tl.functions.users import GetFullUserRequest
from telethon.tl.functions.stories import GetPeerStoriesRequest
from telethon.tl.types import InputPeerUser
import re
from loguru import logger
from src.config import (
    API_HASH,
    API_ID,
    STRING_SESSION
)
import os
from telethon.sessions import StringSession

@dataclass
class ProfileClassifier:
    def __init__(self):
        """
        A class representing a profile analyzer for Telegram users.
        """
        try:
            # Загрузка конфигурации
            with open("./config.yml", "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not all(key in config for key in ["stop_words", "dangerous_words", "spam_words", "whitelist_urls"]):
                    raise ValueError("Отсутствуют обязательные параметры в конфигурации")
                    
                self.path_stop_words = config["stop_words"]
                self.path_dangerous_words = config["dangerous_words"]
                self.path_spam_words = config["spam_words"]
                self.path_whitelist_urls = config["whitelist_urls"]

            # Загрузка списков слов
            self.stop_words = pd.read_csv(self.path_stop_words, sep=";")["stop_words"].tolist()
            self.dangerous_words = pd.read_csv(self.path_dangerous_words, sep=";")["dangerous_words"].tolist()
            self.spam_words = pd.read_csv(self.path_spam_words, sep=";")["spam_words"].tolist()
            self.whitelist_urls = pd.read_csv(self.path_whitelist_urls, sep=";")["whitelist_urls"].tolist()

            self.client = None

        except Exception as e:
            logger.error(f"Ошибка при инициализации ProfileClassifier: {e}")
            raise

    async def start(self):
        """
        Асинхронная инициализация клиента с использованием string session
        """
        try:
            if self.client is None:
                # Создаем клиент с использованием string session
                self.client = TelegramClient(
                    StringSession(STRING_SESSION),
                    API_ID,
                    API_HASH
                )
                
                # Подключаемся к Telegram
                await self.client.connect()
                
                # Проверяем авторизацию
                if not await self.client.is_user_authorized():
                    logger.error("Session is not authorized")
                    raise Exception("User authorization failed. Please check your string session.")
                
                logger.info("Successfully connected and authorized using string session")
                
        except Exception as e:
            logger.error(f"Failed to start Telegram client: {e}")
            raise

    async def _get_user_info(self, user_id: int):
        """
        Получение информации о пользователе с повторными попытками
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Проверяем подключение
                if not self.client.is_connected():
                    await self.client.connect()
                    await asyncio.sleep(1)  # Даем время на подключение

                # Получаем информацию о пользователе
                user = await self.client(GetFullUserRequest(user_id))
                return user
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed to get user info: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)  # Пауза перед следующей попыткой

    async def analyze_profile(self, user_id: int) -> Dict:
        """
        Анализ профиля пользователя
        """
        try:
            await self.start()
            
            logger.info(f"Starting profile analysis for user {user_id}")
            user_id = int(user_id)
            
            try:
                # Получаем сущность пользователя
                user_entity = await self._get_user_entity(user_id)
                
                # Получаем полную информацию
                user = await self.client(GetFullUserRequest(user_entity))
                logger.info(f"Successfully got user info for {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to get user info: {e}")
                return {
                    'error': str(e),
                    'overall_score': 0.0,
                    'features': f"Не удалось получить информацию о пользователе: {str(e)}"
                }

            total_score = 0.0
            features = ""
            results = {}

            # Выполняем проверки
            checks = [
                {"name": "username", "method": self._check_username, "weight": 1.0},
                {"name": "username_emoji", "method": self._check_username_emoji, "weight": 0.8},
                {"name": "bio_words", "method": self._check_bio_words, "weight": 1.0},
                {"name": "bio_links", "method": self._check_bio_links, "weight": 1.0},
                {"name": "bio_emoji", "method": self._check_bio_emoji, "weight": 0.8},
                {"name": "stories_words", "method": self._check_stories_words, "weight": 1.0},
                {"name": "stories_links", "method": self._check_stories_links, "weight": 1.0},
                {"name": "stories_emoji", "method": self._check_stories_emoji, "weight": 0.8}
            ]

            # Выполняем проверки последовательно
            for check in checks:
                try:
                    result = await check["method"](user_entity.id)  # Используем id из полученной сущности
                    if isinstance(result, dict):
                        score = result.get('score', 0) * check["weight"]
                        total_score += score
                        
                        if 'feature' in result and result['feature']:
                            features += f"[{check['name']}] {result['feature']}\n"
                        
                        results[check['name']] = result
                        logger.info(f"Check {check['name']} completed with score {score}")
                except Exception as e:
                    logger.error(f"Error in {check['name']}: {e}")
                    continue

            results['overall_score'] = min(total_score, 1.0)
            results['features'] = features if features else "Подозрительных особенностей не обнаружено"
            
            logger.info(f"Profile analysis completed for user {user_id}. Score: {total_score}")
            return results

        except Exception as e:
            logger.error(f"Error analyzing profile {user_id}: {e}", exc_info=True)
            return {
                'error': str(e),
                'overall_score': 0.0,
                'features': f"Ошибка при анализе профиля: {str(e)}"
            }
        finally:
            try:
                if self.client and self.client.is_connected():
                    await self.client.disconnect()
                    logger.info("Telegram client disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting client: {e}")

    def predict(self, X):
        """
        Predicts the scores for the given input using the trained model.

        Parameters:
            X (pandas DataFrame): The input data to predict scores for.

        Returns:
            numpy array: An array of predicted scores for the input data.
        """
        logger.info("Predicting...")
        if X.empty:
            logger.warning("Empty DataFrame provided")
            return 0.0, ""
            
        total_score = 0.0
        name_features = ""
        for rule in self.rules:
            temp_score, temp_name_features = rule["check"](X.iloc[0, :])
            total_score += temp_score
            name_features += temp_name_features
        total_score_normalized = self._normalize_score(total_score, threshold=0.8)

        return total_score_normalized, name_features

    def _normalize_score(self, score, threshold = 0.4):
        """
        Normalize the score to a range from 0 to 1 using a threshold value.

        Parameters:
            score (float): The input score.
            threshold (float): The threshold value above which the score is considered maximum (1).

        Returns:
            float: The normalized score.
        """
        if score >= threshold:
            normalized_score = 1
        else:
            normalized_score = 0

        return normalized_score


    def _load_word_list(self, path: str, column_name: str) -> List[str]:
        """
        Загрузка списка слов из CSV файла
        """
        try:
            df = pd.read_csv(path, sep=";")
            return df[column_name].dropna().tolist()
        except Exception as e:
            print(f"Ошибка при загрузке {column_name}: {e}")
            return []

    async def _check_username(self, user_id: int) -> Dict:
        """
        Проверка реального имени пользователя (first_name + last_name)
        """
        try:
            logger.info(f"Starting name check for user_id: {user_id}")
            
            # Получаем сущность пользователя напрямую
            try:
                user_entity = await self.client.get_entity(user_id)
                logger.info(f"Got user entity: {user_entity}")
                
                # Получаем имя и фамилию напрямую из сущности
                first_name = getattr(user_entity, 'first_name', '') or ''
                last_name = getattr(user_entity, 'last_name', '') or ''
                full_name = f"{first_name} {last_name}".strip()
                
                logger.info(f"Got full name directly: '{full_name}'")
                
            except Exception as e:
                logger.error(f"Failed to get user entity: {e}")
                return {
                    'score': 0.0,
                    'error': f"Не удалось получить данные пользователя: {str(e)}"
                }
            
            if not full_name:
                logger.warning(f"No name found for user {user_id}")
                return {
                    'score': 0.0,
                    'matches': [],
                    'feature': 'Имя пользователя отсутствует'
                }
            
            score = 0.0
            matches = []
            feature = ""
            threshold = 50  # Порог совпадения
            
            # Проверка на спам слова
            for word in self.spam_words:
                if fuzz.token_set_ratio(word.lower(), full_name.lower()) >= threshold:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В имени "{full_name}" найдено спам слово: "{word}"\n'
                    logger.info(f"Found spam word '{word}' in name '{full_name}'")
            
            # Проверка на опасные слова
            for word in self.dangerous_words:
                if fuzz.token_set_ratio(word.lower(), full_name.lower()) >= threshold:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В имени "{full_name}" найдено опасное слово: "{word}"\n'
                    logger.info(f"Found dangerous word '{word}' in name '{full_name}'")
            
            # Проверка на стоп слова
            for word in self.stop_words:
                if fuzz.token_set_ratio(word.lower(), full_name.lower()) >= threshold:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В имени "{full_name}" найдено стоп слово: "{word}"\n'
                    logger.info(f"Found stop word '{word}' in name '{full_name}'")

            return {
                'score': min(score, 1.0),
                'matches': matches,
                'full_name': full_name,
                'first_name': first_name,
                'last_name': last_name,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Error in name check for user {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    async def _check_username_emoji(self, user_id: int) -> Dict:
        """
        Проверка имени пользователя на наличие подозрительных эмодзи
        
        Args:
            user_id (int): ID пользователя Telegram
            
        Returns:
            Dict: Результаты проверки
        """
        try:
            user = await self.client(GetFullUserRequest(user_id))
            first_name = user.user.first_name or ""
            last_name = user.user.last_name or ""
            
            full_name = f"{first_name} {last_name}".strip()
            
            result = await self._check_emoji_all(full_name)
            if result['suspicious_emojis']:
                result['feature'] = f"[Name] {result['feature']}"
            
            return result

        except Exception as e:
            logger.error(f"Ошибка при проверке имени на эмодзи {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_bio_words(self, user_id: int) -> Dict:
        """
        Проверка био пользователя на наличие подозрительных слов
        """
        try:
            user = await self.client(GetFullUserRequest(user_id))
            bio = user.full_user.about or ""
            
            score = 0.0
            matches = []
            feature = ""
            
            # Проверка на опасные слова
            for word in self.dangerous_words:
                if fuzz.token_set_ratio(word.lower(), bio.lower()) >= 77:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В био содержится опасное слово: "{word}"\n'
            
            for word in self.spam_words:
                if fuzz.token_set_ratio(word.lower(), bio.lower()) >= 77:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В био содержится спам слово: "{word}"\n'
            
            for word in self.stop_words:
                if fuzz.token_set_ratio(word.lower(), bio.lower()) >= 77:
                    score += 0.3
                    matches.append(word)
                    feature += f'[+0.3] - В био содержится стоп слово: "{word}"\n'
            
            return {
                'score': min(score, 1.0),
                'matches': matches,
                'bio': bio,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Ошибка при проверке био на слова {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_bio_links(self, user_id: int) -> Dict:
        """
        Проверка био пользователя на наличие ссылок
        """
        try:
            user = await self.client(GetFullUserRequest(user_id))
            bio = user.full_user.about or ""
            
            score = 0.0
            feature = ""
            
            # Проверка на ссылки
            link_pattern = re.compile(
                r"https?:\/\/(?:t\.me|telegra\.ph)\/[^\s]+|"  # обычные http и https ссылки
                r"@[\w\d_]+|"  # @username формат
                r"t\.me/\S+|"  # t.me ссылки
                r"t\.me/joinchat/\S+|"  # t.me ссылки на группы
                r"telegra\.ph/\S+"  # telegraph ссылки
            )
            
            links = link_pattern.findall(bio)
            if links:
                if bio.strip() == links[0]:
                    score += 0.3
                    feature += "[+0.3] - В био содержится только telegram ссылка\n"
                elif len(links) >= 2:
                    score += 0.3
                    feature += "[+0.3] - В био содержится много telegram ссылок\n"
                else:
                    score += 0.15
                    feature += "[+0.15] - В био содержится telegram ссылка\n"
            
            return {
                'score': min(score, 1.0),
                'links': links,
                'bio': bio,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Ошибка при проверке био на ссылки {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_stories_words(self, user_id: int) -> Dict:
        """
        Проверка историй пользователя на наличие подозрительных слов
        """
        try:
            logger.info(f"Checking stories for user {user_id}")
            
            # Получаем информацию о пользователе
            user = await self.client.get_entity(user_id)
            input_peer = InputPeerUser(user.id, user.access_hash)
            
            # Получаем истории пользователя
            stories = await self.client(GetPeerStoriesRequest(peer=input_peer))
            
            if stories.stories and stories.stories.stories:
                logger.info(f"Found {len(stories.stories.stories)} stories for user {user_id}")
            else:
                logger.info(f"No stories found for user {user_id}")
            
            score = 0.0
            matches = []
            feature = ""
            
            if stories.stories and stories.stories.stories:
                for story in stories.stories.stories:
                    # Проверка подписей к историям
                    if hasattr(story, 'caption'):
                        story_text = story.caption
                        
                        # Проверка на опасные слова в подписи
                        for word in self.dangerous_words:
                            if fuzz.token_set_ratio(word.lower(), story_text.lower()) >= 77:
                                score += 0.3
                                matches.append(word)
                                feature += f'[+0.3] - В подписи к истории содержится: "{word}"\n'
            
            return {
                'score': min(score, 1.0),
                'matches': matches,
                'stories_count': len(stories.stories.stories) if stories.stories else 0,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Error checking stories for user {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_stories_links(self, user_id: int) -> Dict:
        """
        Проверка историй пользователя на наличие ссылок
        
        Args:
            user_id (int): ID пользователя Telegram
            
        Returns:
            Dict: Результаты проверки
        """
        try:
            # Получаем информацию о пользователе
            user = await self.client.get_entity(user_id)
            input_peer = InputPeerUser(user.id, user.access_hash)
            
            # Получаем истории пользователя
            stories = await self.client(GetPeerStoriesRequest(peer=input_peer))
            
            score = 0.0
            urls = []
            feature = ""
            
            # Паттерн для поиска telegram ссылок
            link_pattern = re.compile(
                r"https?:\/\/(?:t\.me|telegra\.ph)\/[^\s]+|"  # обычные http и https ссылки
                r"@[\w\d_]+|"  # @username формат
                r"t\.me/\S+|"  # t.me ссылки
                r"t\.me/joinchat/\S+|"  # t.me ссылки на группы
                r"telegra\.ph/\S+"  # telegraph ссылки
            )
            
            if stories.stories and stories.stories.stories:
                for story in stories.stories.stories:
                    # Проверка веб-страниц в историях
                    if hasattr(story, 'media') and hasattr(story.media, 'webpage'):
                        urls.append(story.media.webpage.url)
                        score += 0.2
                        feature += "[+0.2] - История содержит внешнюю ссылку\n"
                    
                    # Проверка подписей к историям на наличие ссылок
                    if hasattr(story, 'caption'):
                        story_text = story.caption
                        links = link_pattern.findall(story_text)
                        
                        if links:
                            if story_text.strip() == links[0]:
                                score += 0.3
                                feature += "[+0.3] - История содержит только telegram ссылку\n"
                            elif len(links) >= 2:
                                score += 0.3
                                feature += "[+0.3] - История содержит много telegram ссылок\n"
                            else:
                                score += 0.15
                                feature += "[+0.15] - История содержит telegram ссылку\n"
                            urls.extend(links)
            
            return {
                'score': min(score, 1.0),
                'urls': list(set(urls)),  # Удаляем дубликаты ссылок
                'stories_count': len(stories.stories.stories) if stories.stories else 0,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Ошибка при проверке историй на ссылки {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_bio_emoji(self, user_id: int) -> Dict:
        """
        Проверка био пользователя на наличие подозрительных эмодзи
        """
        try:
            user = await self.client(GetFullUserRequest(user_id))
            bio = user.full_user.about or ""
            
            result = await self._check_emoji_all(bio)
            if result['suspicious_emojis']:
                result['feature'] = f"[Bio] {result['feature']}"
            
            return result

        except Exception as e:
            logger.error(f"Ошибка при проверке био на эмодзи {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_stories_emoji(self, user_id: int) -> Dict:
        """
        Проверка историй пользователя на наличие подозрительных эмодзи
        """
        try:
            user = await self.client.get_entity(user_id)
            input_peer = InputPeerUser(user.id, user.access_hash)
            stories = await self.client(GetPeerStoriesRequest(peer=input_peer))
            
            total_score = 0.0
            all_emojis = []
            feature = ""
            
            if stories.stories and stories.stories.stories:
                for story in stories.stories.stories:
                    if hasattr(story, 'caption'):
                        result = await self._check_emoji_all(story.caption)
                        total_score += result['score']
                        all_emojis.extend(result['suspicious_emojis'])
                        feature += result['feature']
            
            return {
                'score': min(total_score, 1.0),
                'suspicious_emojis': list(set(all_emojis)),
                'stories_count': len(stories.stories.stories) if stories.stories else 0,
                'feature': feature
            }

        except Exception as e:
            logger.error(f"Ошибка при проверке историй на эмодзи {user_id}: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def close(self):
        """
        Закрытие клиента
        """
        try:
            if hasattr(self, 'client'):
                if self.client.is_connected():
                    await self.client.disconnect()
                # Удаляем файл сессии после использования
                session_path = self.client.session.filename
                if os.path.exists(session_path):
                    os.remove(session_path)
                logger.info("Telegram client closed and session file removed")
        except Exception as e:
            logger.error(f"Error closing Telegram client: {e}")

    async def _get_user_entity(self, user_id: int):
        """
        Получение сущности пользователя с несколькими попытками разными методами
        """
        try:
            # Метод 1: Прямое получение
            try:
                return await self.client.get_input_entity(user_id)
            except ValueError:
                pass

            # Метод 2: Через поиск в диалогах
            try:
                async for dialog in self.client.iter_dialogs():
                    if dialog.entity.id == user_id:
                        return dialog.entity
            except Exception:
                pass

            # Метод 3: Через API
            try:
                return await self.client.get_entity(user_id)
            except Exception:
                pass

            # Метод 4: Через InputPeerUser
            try:
                return InputPeerUser(user_id, 0)
            except Exception:
                pass

            raise ValueError(f"Could not find user with ID {user_id}")

        except Exception as e:
            logger.error(f"Failed to get user entity: {e}")
            raise

    async def _check_emoji_all(self, text: str) -> Dict:
        """
        Общий метод проверки текста на подозрительные эмодзи
        """
        score = 0.0
        feature = ""
        
        # Unicode кодовые точки для подозрительных эмодзи
        emojis_code = [
            "\U00002757",  # Red Exclamation Mark
            "\U00002753",  # Question Mark Ornament
            "\U0001F4A6",  # Sweat Droplets
            "\U0001F4B5",  # Dollar Banknote
            "\U0001F4B0",  # Money Bag
            "\U0001F4A7",  # Droplet
            "\U0001F346",  # Eggplant
            "\U0001F34C",  # Banana
            "\U0001F351",  # Peach
            "\U0001F353",  # Strawberry
            "\U0001F352",  # Cherries
            "\U0001F608",  # Smiling Face with Horns
            "\U00002705",  # White Heavy Check Mark
            "\U0001F381",  # Wrapped Present
            "\U0001F48E",  # Gem Stone
            "\U0001F911",  # Money-Mouth Face
            "\U00002728",  # Sparkles
            "\U0001F6A8",  # Police Cars Revolving Light
        ]

        emoji_pattern = re.compile("|".join(emojis_code))
        found_emojis = emoji_pattern.findall(text)

        if found_emojis:
            score = 0.3
            feature = f"[+0.3] - Найдены подозрительные эмодзи ({', '.join(found_emojis)})\n"
        
        return {
            'score': score,
            'suspicious_emojis': found_emojis,
            'feature': feature
        }








