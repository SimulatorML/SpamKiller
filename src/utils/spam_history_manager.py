import json
from datetime import datetime, timedelta
from typing import Dict, List
import os
from loguru import logger
# from src.config import WHITELIST_USERS
# from src.utils.commands import add_user_to_whitelist


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

    def add_message(self, user_id: int, spam_category: str, score: float) -> Dict:
        """Добавляет новое сообщение в историю"""
        history = self._load_history()
        str_user_id = str(user_id)
        if str_user_id not in history:
            history[str_user_id] = []

        message_data = {
            "timestamp": datetime.now().isoformat(),
            "spam_category": spam_category,
            "score": score,
            "action": None,
        }

        history[str(user_id)].append(message_data)
        if self._check_whitelist_eligibility(history[str_user_id]):
            message_data["action"] = "whitelist"
        self._save_history(history)

        return self._check_user_status(user_id, history[str_user_id])

    def _check_user_status(self, user_id: int, user_history: List[Dict]) -> Dict:
        """Проверяет статус пользователя на основе истории сообщений"""

        result = {
            "action": None,  # "ban", "delete", "whitelist", None
            "reason": None,
        }

        # Проверка на definite_spam
        if user_history[-1]["spam_category"] == "definite_spam":
            result["action"] = "ban"
            result["reason"] = "definite_spam detected"
            return result

        # Проверка на likely_spam с историей
        if user_history[-1]["spam_category"] == "likely_spam":
            result["action"] = "delete"  # delete at 1st occurrence

            # last_timestamp = datetime.fromisoformat(user_history[-1]["timestamp"])

            current_time = datetime.fromisoformat(user_history[-1]["timestamp"])
            likely_spam_count = 1  # count of likely_spam at 1st occurrence

            for msg in reversed(user_history[:-1]):
                msg_time = datetime.fromisoformat(msg["timestamp"])
                if (current_time - msg_time) <= timedelta(days=7):
                    if msg["spam_category"] == "likely_spam":
                        likely_spam_count += 1
                        if likely_spam_count >= 2:
                            result["action"] = "ban"
                            result["reason"] = "repeated_likely_spam within 7 days"
                            break

                    # prev_timestamp = datetime.fromisoformat(msg["timestamp"])
                    # if last_timestamp - prev_timestamp < timedelta(days=7):
                    #     result["action"] = "ban"
                    #     result["reason"] = "repeated_likely_spam"
                    #     break
                else:
                    break

        # Проверка на добавление в белый список(3 последних not_spam подряд)
        if len(user_history) >= 3:
            last_three = user_history[-3:]
            if all(msg["spam_category"] == "not_spam" for msg in last_three):
                # if user_id not in WHITELIST_USERS: #
                result["action"] = "whitelist"
                result["reason"] = "three_consecutive_not_spam"

        return result

    def is_in_whitelist(self, user_id: int) -> bool:
        """Проверяет, находится ли пользователь в whitelist"""
        history = self._load_history()
        str_user_id = str(user_id)

        if str_user_id not in history:
            return False

        user_history = history[str_user_id]
        return any(action.get("action") == "whitelist" for action in user_history)

    def _check_whitelist_eligibility(self, user_history: List[Dict]) -> bool:
        """Проверяет возможность добавления в whitelist"""
        if len(user_history) >= 2:  # Проверяем предыдущие сообщения
            last_three = user_history[-3:]
            return all(msg["spam_category"] == "not_spam" for msg in last_three)
        return False
