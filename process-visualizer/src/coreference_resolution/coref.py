# --- START OF FILE coref.py ---

import os
import torch
from fastcoref import FCoref # Используем FCoref из fastcoref
import traceback # Для вывода стека ошибок

# --- Функции чтения файлов остаются без изменений (но ignore_words здесь больше не используется) ---
def read_file(file_path):
    """
    Reads a file and returns a list of strings, where each string is a line in the file.
    Args:
        file_path (str): Path to the file.
    Returns:
        (list): List of strings.
    """
    # Добавим проверку существования файла для надежности
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}.")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]
        return lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


# --- Загрузка модели fastcoref ---

# Выбираем устройство (GPU, если доступно, иначе CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Загружаем модель fastcoref ОДИН РАЗ при импорте модуля.
# Это делает ее доступной для импорта в process_bpmn_data.py
try:
    # Используем модель, которая точно работала на предыдущих шагах
    coref_model = FCoref('biu-nlp/f-coref', device=device)
    print("FastCoref model loaded successfully.")
except Exception as e:
    print(f"{'='*20}\nCRITICAL ERROR: Failed to load FCoref model!\n{'='*20}")
    print(f"Device selected: {device}")
    print(f"Error details: {e}")
    traceback.print_exc()
    # В случае ошибки загрузки модели, присваиваем None, чтобы проверки в другом файле сработали
    coref_model = None
    # Можно также возбудить исключение, чтобы прервать выполнение программы
    # raise RuntimeError("Failed to load FCoref model, cannot continue.") from e


# --- ОСНОВНАЯ ФУНКЦИЯ: Получение информации о кореференции ---

def get_coref_info(text: str, print_clusters: bool = False) -> dict | None:
    """
    Получает информацию о кореференции (исходный текст и кластеры) для заданного текста.
    Не изменяет текст, а возвращает структуру данных.
    Args:
        text (str): Входной текст для анализа.
        print_clusters (bool): Если True, печатает найденные кластеры в консоль.
    Returns:
        (dict | None): Словарь с ключами 'text', 'clusters_str', 'clusters_idx'
                       или None в случае ошибки или отсутствия предсказаний.
                       'text': исходный входной текст.
                       'clusters_str': список списков строк-упоминаний.
                       'clusters_idx': список списков кортежей индексов (start_token, end_token).
    """
    # Проверяем, загрузилась ли модель
    if coref_model is None:
        print("ERROR: Coref model was not loaded. Skipping coreference resolution.")
        return None

    # Проверка входного текста
    if not text or not isinstance(text, str) or len(text.split()) < 2:
         print("Skipping coref for short/invalid text.")
         # Возвращаем словарь с исходным текстом, но пустыми кластерами,
         # чтобы вызывающий код мог единообразно обработать
         return {'text': text, 'clusters_str': [], 'clusters_idx': []}

    try:
        # fastcoref ожидает список текстов, даже если он один
        preds = coref_model.predict(texts=[text])

        # Проверяем, есть ли результат
        if not preds:
            print("WARN: Coref model returned empty predictions list.")
            return {'text': text, 'clusters_str': [], 'clusters_idx': []}

        result_obj = preds[0]

        # Получаем кластеры в разных форматах
        clusters_str = result_obj.get_clusters(as_strings=True)
        clusters_idx = result_obj.get_clusters(as_strings=False) # Может пригодиться для точного маппинга

        if print_clusters:
            print("\nCoreference Clusters (strings):")
            if clusters_str:
                for cluster in clusters_str:
                     print(f"- {cluster}")
            else:
                print("- No clusters found.")

        # Возвращаем словарь с результатами
        return {
            'text': result_obj.text, # Исходный текст, как его видела модель
            'clusters_str': clusters_str,
            'clusters_idx': clusters_idx
        }

    except Exception as e:
        print(f"ERROR during coreference prediction: {e}")
        traceback.print_exc()
        # В случае ошибки возвращаем None, чтобы сигнализировать о проблеме
        return None


# --- Функция пакетной обработки закомментирована ---
# Если она вам нужна, ее нужно будет адаптировать аналогично get_coref_info,
# чтобы она возвращала список словарей с информацией, а не измененные тексты.
# Также нужно будет удалить из нее фильтрацию ignore_words.

# def batch_resolve_references(input_folder: str, output_folder: str):
#     """
#     Resolves coreferences in all .txt files in a folder using batch prediction.
#     Args:
#         input_folder (str): Path to the input folder
#         output_folder (str): Path to the output folder
#     """
#     # ... (этот код требует адаптации) ...
#     pass

# --- Функция чтения ignore_words остается, но больше не используется в этом файле ---
# Она может быть полезна в process_bpmn_data.py, если вы решите применить
# фильтрацию ПОСЛЕ того, как получите финальные имена агентов/задач.
def read_ignore_words(file_path="src/coreference_resolution/ignore_words.txt"):
    """
    Reads a list of words to ignore.
    Args:
        file_path (str): Path to the ignore words file.
    Returns:
        (set): Set of ignore words (lowercase).
    """
    if not os.path.exists(file_path):
        print(f"Warning: Ignore words file not found at {file_path}. No words will be ignored.")
        return set()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            ignore_words = {line.strip().lower() for line in file if line.strip()}
        print(f"Loaded {len(ignore_words)} ignore words from {file_path}.")
        return ignore_words
    except Exception as e:
        print(f"Error reading ignore words file {file_path}: {e}")
        return set()

# Загрузка ignore_words при импорте, чтобы они были доступны для импорта, если нужны
# ignore_words = read_ignore_words()

# --- END OF FILE coref.py ---