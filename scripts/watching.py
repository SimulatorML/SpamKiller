import os
import time
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class JSONFileHandler(FileSystemEventHandler):
    def __init__(self, json_file_paths, scripts_directory, data_directory):
        """
        Initializes an instance of the class with the given `json_file_paths`, `scripts_directory`, and `data_directory`.

        :param json_file_paths: A list of strings containing the paths to the JSON files.
        :type json_file_paths: list
        :param scripts_directory: A string representing the path to the directory containing the scripts.
        :type scripts_directory: str
        :param data_directory: A string representing the path to the directory containing the data.
        :type data_directory: str
        """
        super().__init__()
        self.json_file_paths = json_file_paths
        self.last_modified = {path: os.path.getmtime(path) for path in json_file_paths}
        self.scripts_directory = scripts_directory
        self.data_directory = data_directory

    def on_modified(self, event):
        """
        Executes a series of actions when a file is modified.

        Args:
            self: An instance of the class.
            event: The event that triggered the function.

        Returns:
            None
        """
        if event.src_path in self.json_file_paths and not event.is_directory:
            current_modified = os.path.getmtime(event.src_path)
            if current_modified > self.last_modified[event.src_path]:
                self.last_modified[event.src_path] = current_modified
                sys.path.append(self.data_directory)
                subprocess.run(
                    [
                        "python",
                        os.path.join(self.scripts_directory, "data_preprocessing.py"),
                    ],
                    cwd=os.getcwd(),
                )
                sys.path.remove(self.data_directory)


data_directory = "src/data"
scripts_directory = "scripts"

json_file_paths = [
    os.path.join("data", "text_spam_dataset", "result_not_spam.json"),
    os.path.join("data", "text_spam_dataset", "result_spam.json"),
]

event_handler = JSONFileHandler(json_file_paths, scripts_directory, data_directory)
observer = Observer()
for path in json_file_paths:
    observer.schedule(event_handler, path=os.path.dirname(path), recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    observer.join()
