import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from loguru import logger
from src.config import WHITELIST_USERS
from src.utils.commands import add_user_to_whitelist


class SpamHistoryManager:
    def __init__(self, history_file: str = "data/dataset/spam_history.json"):
        self.history_file = history_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Создает файл истории, если он не существует"""
        if not os.path.exists(self.history_file):
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump({}, f)

    def _load_history(self) -> Dict:
        """Загружает историю из файла"""
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Ошибка чтения {self.history_file}")
            return {}

    def _save_history(self, history: Dict):
        """Сохраняет историю в файл"""
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def add_message(self, user_id: int, spam_category: str, score: float):
        """Добавляет новое сообщение в историю"""
        history = self._load_history()

        if str(user_id) not in history:
            history[str(user_id)] = []

        message_data = {
            "timestamp": datetime.now().isoformat(),
            "spam_category": spam_category,
            "score": score,
        }

        history[str(user_id)].append(message_data)
        self._save_history(history)

        return self._check_user_status(user_id, history[str(user_id)])

    def _check_user_status(self, user_id: int, user_history: List[Dict]) -> Dict:
        """Проверяет статус пользователя на основе истории сообщений"""
        result = {
            "action": None,  # "ban", "delete", "whitelist", None
            "reason": None,
        }

        # Проверка на definite_spam
        if user_history[-1]["spam_category"] == "definite_spam":
            result["action"] = "ban"
            result["reason"] = "definite_spam"
            return result

        # Проверка на likely_spam с историей
        if user_history[-1]["spam_category"] == "likely_spam":
            result["action"] = "delete"
            last_timestamp = datetime.fromisoformat(user_history[-1]["timestamp"])

            for msg in reversed(user_history[:-1]):
                if msg["spam_category"] == "likely_spam":
                    prev_timestamp = datetime.fromisoformat(msg["timestamp"])
                    if last_timestamp - prev_timestamp < timedelta(days=7):
                        result["action"] = "ban"
                        result["reason"] = "repeated_likely_spam"
                        break

        # Проверка на добавление в белый список
        if len(user_history) >= 3:
            last_three = user_history[-3:]
            if all(msg["spam_category"] == "not_spam" for msg in last_three):
                if user_id not in WHITELIST_USERS:
                    result["action"] = "whitelist"
                    result["reason"] = "three_consecutive_not_spam"

        return result
