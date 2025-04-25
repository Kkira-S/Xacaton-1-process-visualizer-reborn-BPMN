# --- START OF FILE logging_utils.py ---
import json
import os
import shutil
from colorama import Fore, init # Добавим colorama и сюда для вывода

init(autoreset=True)

LOG_DIR = "./output_logs" # Определим константу для директории логов

def clear_folder(folder_path: str):
    """Deletes and recreates a folder."""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' cleared.")
        os.makedirs(folder_path, exist_ok=True) # exist_ok=True на случай, если папки не было
        # print(f"Folder '{folder_path}' created.") # Можно раскомментировать для отладки
    except OSError as e:
        print(f"{Fore.RED}Error clearing or creating folder {folder_path}: {e}{Fore.RESET}")

# --- ИЗМЕНЕННАЯ ФУНКЦИЯ ---
def write_to_file(filepath: str, data): # Переименовали filename в filepath для ясности
    """Writes data to a JSON file at the specified full path."""
    try:
        # --- ДОБАВЛЕНО: Убедимся, что директория существует ---
        dir_name = os.path.dirname(filepath)
        # Проверяем, что dir_name не пустой (если файл в текущей директории)
        # и что директория не существует
        if dir_name and not os.path.exists(dir_name):
            print(f"Creating directory: {dir_name}")
            os.makedirs(dir_name, exist_ok=True) # exist_ok=True безопасен
        # --- КОНЕЦ ДОБАВЛЕНИЯ ---

        # Используем переданный filepath НАПРЯМУЮ
        with open(filepath, "w", encoding='utf-8') as file: # Добавлен encoding='utf-8'
            json.dump(data, file, indent=4, ensure_ascii=False) # Добавлен ensure_ascii=False для не-латинских символов
        # print(f"Data successfully written to {filepath}") # Опциональное сообщение об успехе

    except OSError as e:
        # Выводим ошибку, если не удалось создать директорию или записать файл
        print(f"{Fore.RED}Error creating directory or writing file {filepath}: {e}{Fore.RESET}")
    except TypeError as e:
        # Выводим ошибку, если данные не сериализуются в JSON
        print(f"{Fore.RED}Error serializing data for {filepath}. Check data structure. Error: {e}{Fore.RESET}")
    except Exception as e:
        # Ловим другие возможные ошибки
        print(f"{Fore.RED}An unexpected error occurred in write_to_file for {filepath}: {e}{Fore.RESET}")
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

# --- END OF FILE logging_utils.py ---