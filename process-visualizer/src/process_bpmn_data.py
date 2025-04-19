# --- START OF FILE process_bpmn_data.py ---

import json
import re
import traceback # Добавлено для вывода ошибок

import requests
import spacy
from colorama import Fore
from spacy.matcher import Matcher
from thefuzz import fuzz
import deepseek_prompts as prompts
# ----- ИЗМЕНЕННЫЕ ИМПОРТЫ -----
from coreference_resolution.coref import get_coref_info, coref_model # Импортируем новую функцию и модель
# from coreference_resolution.coref import resolve_references # Старая функция больше не нужна
# ----- КОНЕЦ ИЗМЕНЕННЫХ ИМПОРТОВ -----
from create_bpmn_structure import create_bpmn_structure
from graph_generator import GraphGenerator
from logging_utils import clear_folder, write_to_file

# --- Константы и функции до _resolve_agent_mention без изменений ---
BPMN_INFORMATION_EXTRACTION_ENDPOINT = "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction-v2"
ZERO_SHOT_CLASSIFICATION_ENDPOINT = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)
HF_API_TOKEN = "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" # Рекомендуется вынести в переменные окружения

# --- spaCy модели - лучше загружать один раз ---
try:
    nlp_sm = spacy.load("en_core_web_sm")
    nlp_md = spacy.load("en_core_web_md") # Загружаем обе модели один раз
except OSError as e:
    print(f"{Fore.RED}Error loading spaCy models: {e}{Fore.RESET}")
    print(f"{Fore.YELLOW}Please run 'python -m spacy download en_core_web_sm' and 'python -m spacy download en_core_web_md'{Fore.RESET}")
    # Можно либо выйти, либо продолжить, но функции, использующие spaCy, будут падать
    # exit() # Раскомментировать, если модели критичны для запуска


def get_sentences(text: str) -> list[str]:
    """
    Creates a list of sentences from a given text using the preloaded spaCy model.
    Args:
        text (str): the text to split into sentences
    Returns:
        list: a list of sentences
    """
    if 'nlp_sm' not in globals(): # Проверка, загрузилась ли модель
        print(f"{Fore.RED}spaCy 'en_core_web_sm' model not loaded. Cannot split sentences.{Fore.RESET}")
        return [text] # Возвращаем исходный текст как одно предложение
    doc = nlp_sm(text)
    sentences = [sent.text for sent in doc.sents] # Используем sent.text
    return sentences


def create_sentence_data(text: str) -> list[dict]:
    """
    Создает список словарей, содержащих данные предложения (предложение, начальный индекс, конечный индекс).
    Args:
        text (str): the input text
    Returns:
        list: a list of dictionaries containing the sentence data
    """
    sentences = get_sentences(text)
    start = 0
    sentence_data = []
    current_pos = 0 # Используем для отслеживания позиции в исходном тексте

    for sent_text in sentences:
        # Ищем текущее предложение в тексте начиная с current_pos
        # Это более надежно, чем просто прибавлять длину, т.к. spaCy может менять пробелы
        try:
            start = text.index(sent_text, current_pos)
            end = start + len(sent_text)
            sentence_data.append({"sentence": sent_text, "start": start, "end": end})
            current_pos = end # Обновляем позицию для следующего поиска
        except ValueError:
            # Если предложение не найдено (редко, но возможно из-за нормализации spaCy)
            print(f"{Fore.YELLOW}Warning: Could not accurately find sentence boundaries for: '{sent_text}'. Indices might be approximate.{Fore.RESET}")
            # Приблизительный расчет, как было раньше
            end = start + len(sent_text)
            sentence_data.append({"sentence": sent_text, "start": start, "end": end})
            start = end + 1 # Приблизительное смещение (может быть неточно)
            current_pos = start

    write_to_file("sentence_data.json", sentence_data)
    return sentence_data


def query(payload: dict, endpoint: str) -> dict: # Возвращает dict, а не list
    """
    Отправляет POST-запрос на указанную конечную точку с заданной полезной нагрузкой в формате JSON и возвращает ответ.
    Args:
        payload (dict): the payload to send to the endpoint
        endpoint (str): the endpoint to send the request to
    Returns:
        dict: the response from the endpoint or an error dictionary
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json" # Явно указываем тип контента
    }
    data = json.dumps(payload)
    try:
        response = requests.post(endpoint, data=data, headers=headers, timeout=30) # Добавлен таймаут
        response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)
        return response.json() # Используем response.json()
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}HTTP Request failed: {e}{Fore.RESET}")
        return {"error": f"HTTP Request failed: {str(e)}", "status_code": getattr(e.response, 'status_code', None)}
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}Failed to decode JSON response: {e}{Fore.RESET}")
        # Попытаемся показать часть ответа для отладки
        raw_response = getattr(response, 'text', 'N/A')
        print(f"{Fore.YELLOW}Raw response (partial): {raw_response[:500]}{Fore.RESET}")
        return {"error": f"Failed to decode JSON: {str(e)}", "raw_response": raw_response}
    except Exception as e: # Ловим другие возможные ошибки
        print(f"{Fore.RED}An unexpected error occurred during API query: {e}{Fore.RESET}")
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}


def extract_bpmn_data(text: str) -> list[dict] | None: # Уточнили возвращаемый тип
    """
    Извлекает данные BPMN из описания процесса, вызывая конечную точку модели, размещенную на Hugging Face.
    Args:
        text (str): the process description
    Returns:
        list[dict] | None: model output (list of entities) or None if an error occurred
    """
    print("Extracting BPMN data...\n")
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    data = query(payload, BPMN_INFORMATION_EXTRACTION_ENDPOINT)

    # Проверяем наличие ключа ошибки и является ли результат списком (ожидаемый формат)
    if isinstance(data, dict) and "error" in data:
        print(f"{Fore.RED}Error when extracting BPMN data: {data['error']}{Fore.RESET}")
        if "status_code" in data:
             print(f"Status Code: {data['status_code']}")
        return None
    elif not isinstance(data, list):
        print(f"{Fore.RED}Unexpected response format from BPMN extraction model. Expected list, got {type(data)}.{Fore.RESET}")
        print(f"Response: {data}")
        return None

    write_to_file("model_output.json", data)
    return data


def fix_bpmn_data(data: list[dict]) -> list[dict]:
    """
    Если модель, которая извлекает данные BPMN, по какой-либо причине разбивает слово на несколько токенов,
    эта функция исправляет вывод, объединяя токены в одно слово.
    Args:
        data (list): the model output
    Returns:
        list: the model output with the tokens combined into a single word
    """
    # Реализация без изменений, но можно сделать чуть эффективнее
    if not data: # Проверка на пустой список
        return []

    fixed_data = []
    i = 0
    while i < len(data):
        current_entity = data[i]
        # Проверяем следующий элемент только если он существует
        if i + 1 < len(data):
            next_entity = data[i+1]
            # Условие объединения
            if (current_entity["entity_group"] == "TASK"
                and next_entity["entity_group"] == "TASK"
                and current_entity["end"] == next_entity["start"]):

                # print("Fixing BPMN data...") # Опционально для отладки
                # print(f"Combining '{current_entity['word']}' and '{next_entity['word']}'")

                combined_word = current_entity["word"]
                if next_entity["word"].startswith("##"):
                    combined_word += next_entity["word"][2:]
                else:
                    # Добавляем пробел, если следующий токен не начинается с ## (маловероятно для этой модели, но безопасно)
                    # Хотя для имен задач пробел может быть и не нужен
                    combined_word += next_entity["word"] # Или combined_word += " " + next_entity["word"]

                current_entity["word"] = combined_word
                current_entity["end"] = next_entity["end"]
                current_entity["score"] = max(current_entity["score"], next_entity["score"])
                # Добавляем объединенную сущность и пропускаем следующую
                fixed_data.append(current_entity)
                i += 2 # Перескакиваем через объединенный элемент
                continue # Переходим к следующей итерации цикла while

        # Если не объединяли, просто добавляем текущий элемент
        fixed_data.append(current_entity)
        i += 1

    if len(fixed_data) != len(data):
        print("BPMN data fixed (combined TASK tokens).")
        write_to_file("model_output_fixed.json", fixed_data)

    return fixed_data


def classify_process_info(text: str) -> dict | None: # Уточнили возвращаемый тип
    """
    Classifies a PROCESS_INFO entity by calling the model endpoint hosted on Hugging Face.
    Possible classes: PROCESS_START, PROCESS_END, PROCESS_SPLIT, PROCESS_RETURN
    Args:
        text (str): sequence of text classified as process info
    Returns:
        dict | None: model output containing the following keys: "sequence", "labels", "scores", or None on error
    """
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": [
                "beginning", # Начало
                "end",       # Конец
                "split",     # Разделение / Ветвление
                "going back",# Возврат / Цикл
                "continuation" # Продолжение (нейтрально)
            ]
        },
        "options": {"wait_for_model": True},
    }
    data = query(payload, ZERO_SHOT_CLASSIFICATION_ENDPOINT)

    if isinstance(data, dict) and "error" in data:
        print(f"{Fore.RED}Error when classifying PROCESS_INFO entity '{text}': {data['error']}{Fore.RESET}")
        return None
    # Проверка на ожидаемый формат ответа
    elif not isinstance(data, dict) or not all(k in data for k in ["sequence", "labels", "scores"]):
         print(f"{Fore.RED}Unexpected response format from Zero-Shot model for '{text}'.{Fore.RESET}")
         print(f"Response: {data}")
         return None

    return data


def batch_classify_process_info(process_info_entities: list[dict]) -> list[dict]:
    """
    Classifies a list of PROCESS_INFO entities into PROCESS_START, PROCESS_END, PROCESS_SPLIT or PROCESS_RETURN.
    Args:
        process_info_entities (list): a list of PROCESS_INFO entities
    Returns:
        list: a list of PROCESS_INFO entities with the entity_group key updated
    """
    updated_entities = []
    print("Classifying PROCESS_INFO entities...\n")

    process_info_map = { # Используем map для перевода меток
        "beginning": "PROCESS_START",
        "end": "PROCESS_END",
        "split": "PROCESS_SPLIT",
        "going back": "PROCESS_RETURN",
        "continuation": "PROCESS_CONTINUE", # Оставляем как есть или можно игнорировать
    }

    for entity in process_info_entities:
        text = entity["word"]
        classification_result = classify_process_info(text)

        if classification_result:
            # Берем самую вероятную метку
            top_label = classification_result["labels"][0]
            # Обновляем entity_group, если метка известна, иначе оставляем PROCESS_INFO
            entity["entity_group"] = process_info_map.get(top_label, "PROCESS_INFO") # Используем .get с fallback
            entity["classification_score"] = classification_result["scores"][0] # Сохраняем скор классификации
        else:
            # Оставляем как PROCESS_INFO в случае ошибки классификации
            print(f"{Fore.YELLOW}Could not classify '{text}', leaving as PROCESS_INFO.{Fore.RESET}")
            entity["entity_group"] = "PROCESS_INFO" # Убедимся, что тип остается

        updated_entities.append(entity)

    return updated_entities


def extract_entities(type: str, data: list[dict], min_score: float) -> list[dict]:
    """
    Extracts all entities of a given type from the model output
    Args:
        type (str): the type of entity to extract
        data (list): the model output
        min_score (float): the minimum score an entity must have to be extracted
    Returns:
        list: a list of entities of the given type
    """
    if not data: # Проверка на пустой ввод
        return []
    return [
        entity
        for entity in data
        if entity["entity_group"] == type and entity.get("score", 0) > min_score # Используем .get для безопасности
    ]


# --- НОВАЯ ВНУТРЕННЯЯ ФУНКЦИЯ ДЛЯ РАЗРЕШЕНИЯ АГЕНТОВ ---
def _resolve_agent_mention(agent_entity: dict, clusters: list[list[str]] | None) -> tuple[str, bool]:
    """
    Находит репрезентативное упоминание для агента, используя кластеры кореференций.
    Args:
        agent_entity (dict): Словарь сущности AGENT из NER.
        clusters (list[list[str]] | None): Список кластеров (списки строк).
    Returns:
        tuple[str, bool]: Кортеж (resolved_name, is_resolved), где
                          resolved_name - строка с разрешенным именем агента,
                          is_resolved - флаг, показывающий, была ли выполнена замена.
    """
    if not agent_entity or 'word' not in agent_entity:
        return "", False # Возвращаем пустую строку, если сущность некорректна

    agent_word = agent_entity['word']
    resolved_name = agent_word # По умолчанию - само упоминание
    is_resolved = False

    if clusters:
        # Приводим слово агента к нижнему регистру для сравнения без учета регистра
        agent_word_lower = agent_word.lower()
        # Простые местоимения, которые точно нужно разрешать
        pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "theirs"}

        for cluster in clusters:
            # Проверяем, есть ли *точное* слово (с учетом регистра!) агента в кластере
            # ИЛИ есть ли его *форма в нижнем регистре* (для поиска местоимений и т.п.)
            found_in_cluster = agent_word in cluster or agent_word_lower in [c.lower() for c in cluster]

            if found_in_cluster:
                # Нашли кластер. Выбираем репрезентативное упоминание.
                # Логика: Берем первое упоминание, если оно не местоимение.
                # Если первое - местоимение, ищем первое не-местоимение в кластере.
                # Если все - местоимения, берем первое.
                representative_mention = cluster[0] # По умолчанию первое
                if representative_mention.lower() in pronouns and len(cluster) > 1:
                    # Попробуем найти первое не-местоимение
                    first_non_pronoun = next((mention for mention in cluster if mention.lower() not in pronouns), None)
                    if first_non_pronoun:
                        representative_mention = first_non_pronoun
                    # Если не нашли не-местоимение, оставляем первое (которое местоимение)

                # Проверяем, отличается ли найденное репрезентативное имя от исходного
                # и является ли исходное слово местоимением (чтобы не заменять "Customer" на "Customer")
                if agent_word != representative_mention and agent_word_lower in pronouns:
                     resolved_name = representative_mention
                     is_resolved = True
                     # print(f"DEBUG Coref: Resolved '{agent_word}' -> '{resolved_name}'") # Отладка
                # Если агент не местоимение, но мы нашли для него кластер (например, "the client" -> "Customer"),
                # можно тоже разрешить, но осторожно. Пока разрешаем только местоимения.
                # elif agent_word != representative_mention:
                #     resolved_name = representative_mention
                #     is_resolved = True

                break # Выходим, найдя первый подходящий кластер
    return resolved_name, is_resolved
# --- КОНЕЦ ВНУТРЕННЕЙ ФУНКЦИИ ---


def create_agent_task_pairs(
    agents: list[dict],
    tasks: list[dict],
    sentence_data: list[dict],
    clusters: list[list[str]] | None = None, # Принимаем кластеры
) -> list[dict]:
    """
    Объединяет агентов и задачи в пары агент-задача на основе предложения,
    разрешая кореференции для агентов, если предоставлены кластеры.
    Args:
        agents (list): a list of agents (NER results)
        tasks (list): a list of tasks (NER results)
        sentence_data (list): a list of sentence data
        clusters (list[list[str]], optional): Coreference clusters as lists of strings. Defaults to None.
    Returns:
        list: a list of agent-task pairs dictionaries. Agent is now a dict itself.
    """
    # Находим агентов и задачи в предложениях
    agents_in_sentences = [
        {"sentence_idx": i, "agent": agent}
        for agent in agents
        for i, sent in enumerate(sentence_data)
        # Используем строгое включение, чтобы избежать дублирования на границах
        if sent["start"] <= agent["start"] < sent["end"]
    ]
    tasks_in_sentences = [
        {"sentence_idx": i, "task": task}
        for task in tasks
        for i, sent in enumerate(sentence_data)
        if sent["start"] <= task["start"] < sent["end"]
    ]

    # Определяем предложения с несколькими агентами
    multi_agent_sentences_idx = set()
    counts = {}
    for agent_sent in agents_in_sentences:
        idx = agent_sent["sentence_idx"]
        counts[idx] = counts.get(idx, 0) + 1
        if counts[idx] > 1:
            multi_agent_sentences_idx.add(idx)

    # Создаем пары для предложений с одним агентом
    agent_task_pairs = []
    processed_tasks_indices = set() # Чтобы не дублировать задачи

    for agent_sent in agents_in_sentences:
        sent_idx = agent_sent["sentence_idx"]
        if sent_idx in multi_agent_sentences_idx:
            continue # Пропускаем, обработаем позже

        agent_entity = agent_sent['agent']
        resolved_name, _ = _resolve_agent_mention(agent_entity, clusters) # Разрешаем агента

        # Находим все задачи в том же предложении
        tasks_for_this_agent = [
            task_sent['task']
            for task_sent in tasks_in_sentences
            if task_sent['sentence_idx'] == sent_idx
        ]

        for task_entity in tasks_for_this_agent:
            task_tuple = (task_entity['start'], task_entity['end']) # Используем кортеж для set
            if task_tuple not in processed_tasks_indices:
                pair = {
                    "agent": {
                        "original_word": agent_entity['word'],
                        "resolved_word": resolved_name,
                        "entity": agent_entity
                    },
                    "task": task_entity,
                    "sentence_idx": sent_idx
                }
                agent_task_pairs.append(pair)
                processed_tasks_indices.add(task_tuple)

    # Обрабатываем предложения с несколькими агентами
    if multi_agent_sentences_idx:
        multi_agent_task_pairs = handle_multi_agent_sentences(
            agents_in_sentences, tasks_in_sentences, list(multi_agent_sentences_idx), clusters
        )
        # Добавляем только те пары, задачи которых еще не были обработаны
        for pair in multi_agent_task_pairs:
             task_tuple = (pair['task']['start'], pair['task']['end'])
             if task_tuple not in processed_tasks_indices:
                 agent_task_pairs.append(pair)
                 processed_tasks_indices.add(task_tuple)

    # Сортируем по индексу предложения, затем по началу задачи
    agent_task_pairs.sort(key=lambda k: (k["sentence_idx"], k["task"]["start"]))
    return agent_task_pairs


def handle_multi_agent_sentences(
    agents_in_sentences: list[dict],
    tasks_in_sentences: list[dict],
    multi_agent_sentences_idx: list[int],
    clusters: list[list[str]] | None = None,
) -> list[dict]:
    """
    Creates agent-task pairs for sentences that contain multiple agents, resolving agents.
    Tries a simple nearest-agent logic.
    Args:
        agents_in_sentences (list): a list of agents with their sentence indices
        tasks_in_sentences (list): a list of tasks with their sentence indices
        multi_agent_sentences_idx (list): a list of sentence indices that contain multiple agents
        clusters (list[list[str]], optional): Coreference clusters. Defaults to None.
    Returns:
        list: a list of agent-task pairs with resolved agents.
    """
    agent_task_pairs = []

    for idx in multi_agent_sentences_idx:
        # Получаем агентов и задачи только для этого предложения
        agents_in_this_sentence = sorted(
            [agent_sent['agent'] for agent_sent in agents_in_sentences if agent_sent['sentence_idx'] == idx],
            key=lambda x: x['start']
        )
        tasks_in_this_sentence = sorted(
            [task_sent['task'] for task_sent in tasks_in_sentences if task_sent['sentence_idx'] == idx],
            key=lambda x: x['start']
        )

        # --- Простая Эвристика: Ближайший предыдущий агент ---
        # Можно улучшить, используя синтаксический анализ (например, через spaCy)
        last_agent_entity = None
        for task_entity in tasks_in_this_sentence:
            # Находим ближайшего агента, который находится *перед* задачей
            closest_agent = None
            min_distance = float('inf')
            for agent_entity in agents_in_this_sentence:
                if agent_entity['end'] <= task_entity['start']: # Агент должен быть до задачи
                    distance = task_entity['start'] - agent_entity['end']
                    if distance < min_distance:
                        min_distance = distance
                        closest_agent = agent_entity

            # Если нашли ближайшего агента
            if closest_agent:
                resolved_name, _ = _resolve_agent_mention(closest_agent, clusters)
                pair = {
                    "agent": {
                        "original_word": closest_agent['word'],
                        "resolved_word": resolved_name,
                        "entity": closest_agent
                    },
                    "task": task_entity,
                    "sentence_idx": idx,
                }
                agent_task_pairs.append(pair)
            # Если агент перед задачей не найден (маловероятно, но возможно)
            # Можно присвоить предыдущего агента или оставить без агента
            elif last_agent_entity: # Используем последнего известного агента из этого предложения
                 resolved_name, _ = _resolve_agent_mention(last_agent_entity, clusters)
                 pair = {
                    "agent": {
                        "original_word": last_agent_entity['word'],
                        "resolved_word": resolved_name,
                        "entity": last_agent_entity
                    },
                    "task": task_entity,
                    "sentence_idx": idx,
                 }
                 agent_task_pairs.append(pair)
            else:
                 print(f"{Fore.YELLOW}Warning: Could not associate agent for task '{task_entity['word']}' in multi-agent sentence {idx}.{Fore.RESET}")

            # Обновляем последнего виденного агента (если текущий ближе)
            if closest_agent:
                last_agent_entity = closest_agent

    return agent_task_pairs


# --- Функции add_process_end_events, has_parallel_keywords, find_sentences_with_loop_keywords БЕЗ ИЗМЕНЕНИЙ ---
# Они работают либо с текстом (has_parallel_keywords), либо с парами, не трогая структуру агента.

def add_process_end_events(
    agent_task_pairs: list[dict],
    sentences: list[dict],
    process_info_entities: list[dict],
) -> list[dict]:
    """
    Adds process end events to agent-task pairs
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        process_info_entities (list): list of process info entities (already classified)
    Returns:
        list: a list of agent-task pairs with process end events potentially added
    """
    # Находим конечные события и их индексы предложений
    process_end_events_map = {} # Словарь: sentence_idx -> end_event_entity
    for entity in process_info_entities:
        if entity["entity_group"] == "PROCESS_END":
            for i, sent in enumerate(sentences):
                # Проверяем вхождение события в предложение
                if sent["start"] <= entity["start"] < sent["end"]:
                    process_end_events_map[i] = entity
                    break # Одно событие на предложение

    # Добавляем событие к парам в соответствующем предложении
    for pair in agent_task_pairs:
        sent_idx = pair.get("sentence_idx")
        if sent_idx is not None and sent_idx in process_end_events_map:
            # Добавляем только если у пары еще нет конечного события
            if "process_end_event" not in pair:
                 pair["process_end_event"] = process_end_events_map[sent_idx]

    return agent_task_pairs


def has_parallel_keywords(text: str) -> bool:
    """
    Проверяет, содержит ли текст параллельные ключевые слова
    Args:
        text (str): the text to check
    Returns:
        bool: True if the text contains parallel keywords, False otherwise
    """
    if 'nlp_md' not in globals():
         print(f"{Fore.RED}spaCy 'en_core_web_md' model not loaded. Cannot check for parallel keywords.{Fore.RESET}")
         return False
    # Используем предзагруженную модель
    matcher = Matcher(nlp_md.vocab)
    patterns = [
        [{"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "meantime"}],
        [{"LOWER": "at"}, {"LOWER": "the"}, {"LOWER": "same"}, {"LOWER": "time"}],
        [{"LOWER": "meanwhile"}],
        [{"LOWER": "while"}], # Может быть слишком широким, нужно проверить
        [{"LOWER": "in"}, {"LOWER": "parallel"}],
        [{"LOWER": "parallel"}, {"LOWER": "paths"}],
        [{"LOWER": "concurrently"}],
        [{"LOWER": "simultaneously"}],
    ]
    matcher.add("PARALLEL", patterns)
    doc = nlp_md(text)
    matches = matcher(doc)
    return len(matches) > 0


def find_sentences_with_loop_keywords(sentences: list[dict]) -> list[dict]:
    """
    Возвращает предложения, содержащие ключевые слова цикла
    Args:
        sentences (list): list of sentence data
    Returns:
        list: list of sentences that contain loop keywords
    """
    if 'nlp_md' not in globals():
         print(f"{Fore.RED}spaCy 'en_core_web_md' model not loaded. Cannot check for loop keywords.{Fore.RESET}")
         return []
    # Используем предзагруженную модель
    matcher = Matcher(nlp_md.vocab)
    patterns = [
        [{"LOWER": "again"}],
        # Можно добавить другие: repeat, iterate, until <condition>, while <condition>
        [{"LOWER": "repeat"}],
        [{"LOWER": "iterate"}],
        # [{"LOWER": "until"}], # Требует более сложного анализа условия
        # [{"LOWER": "while"}], # Уже есть в parallel, может конфликтовать
    ]
    matcher.add("LOOP", patterns)
    detected_sentences = [
        sent for sent in sentences if len(matcher(nlp_md(sent["sentence"]))) > 0
    ]
    return detected_sentences


# --- АДАПТАЦИЯ add_task_ids и add_loops ---
# Они работают с парами, но add_loops удаляет агента и задачу.
# Нужно убедиться, что это не ломает логику дальше, если агент нужен.

def add_task_ids(
    agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]
) -> list[dict]:
    """
    Добавляет идентификаторы задач ('task_id') к задачам, которые не содержатся в предложении с ключевым словом loop.
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        loop_sentences (list): list of sentences that contain loop keywords
    Returns:
        list: a list of agent-task pairs with task ids potentially added
    """
    task_id_counter = 0
    loop_sentence_indices = {sent['start'] for sent in loop_sentences} # Используем set для быстрой проверки

    for pair in agent_task_pairs:
        # Проверяем, есть ли задача в паре (циклы могут ее удалить)
        if "task" in pair:
            task = pair["task"]
            task_in_loop = False
            # Проверяем, попадает ли задача в предложение с циклом
            for sent in sentences:
                if sent["start"] <= task["start"] < sent["end"]:
                    if sent['start'] in loop_sentence_indices:
                        task_in_loop = True
                        break # Нашли предложение с циклом, дальше не ищем

            # Если задача НЕ в предложении с циклом, присваиваем ID
            if not task_in_loop:
                pair["task"]["task_id"] = f"T{task_id_counter}"
                task_id_counter += 1
        # Если в паре нет ключа "task" (например, это уже обработанный цикл), пропускаем
        # else:
        #     print(f"DEBUG: Skipping task ID assignment for pair without 'task': {pair}")

    return agent_task_pairs


def add_loops(
    agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]
) -> list[dict]:
    """
    Добавляет ключи 'go_to' к элементам, представляющим конец цикла, указывая на ID задачи, к которой нужно вернуться.
    Модифицирует структуру элементов цикла, удаляя 'agent' и 'task'.
    Args:
        agent_task_pairs (list): a list of agent-task pairs (некоторые могут быть уже модифицированы add_task_ids)
        sentences (list): list of sentence data
        loop_sentences (list): list of sentences that contain loop keywords
    Returns:
        list: a list of agent-task pairs where loop elements are modified
    """
    # Собираем все задачи, которым был присвоен ID (т.е. не в цикле)
    tasks_with_ids = {pair["task"]["task_id"]: pair["task"]
                      for pair in agent_task_pairs if "task" in pair and "task_id" in pair["task"]}

    # Индексы предложений с циклами для быстрой проверки
    loop_sentence_indices = {sent['start'] for sent in loop_sentences}

    processed_pairs = [] # Собираем новый список, чтобы избежать проблем с изменением при итерации
    for pair in agent_task_pairs:
        # Проверяем, есть ли задача в паре
        if "task" in pair:
            task = pair["task"]
            task_sentence_start = -1
            # Находим предложение, в котором находится задача
            for sent in sentences:
                if sent["start"] <= task["start"] < sent["end"]:
                    task_sentence_start = sent['start']
                    break

            # Если задача находится в предложении с циклом
            if task_sentence_start in loop_sentence_indices:
                 # Найти предыдущую задачу, к которой нужно вернуться
                 # Используем find_previous_task (требует список задач с ID)
                 previous_task_to_loop_to = find_previous_task(list(tasks_with_ids.values()), task)

                 if previous_task_to_loop_to and "task_id" in previous_task_to_loop_to:
                     # Создаем новый элемент для цикла
                     loop_element = {
                         "go_to": previous_task_to_loop_to["task_id"],
                         # Сохраняем исходные координаты для возможной сортировки/привязки
                         "start": task["start"],
                         "end": task["end"],
                         "sentence_idx": pair["sentence_idx"],
                         # Можно добавить исходного агента/задачу для информации
                         "original_loop_task": task,
                         "original_loop_agent": pair.get("agent")
                     }
                     processed_pairs.append(loop_element)
                 else:
                     # Не удалось найти предыдущую задачу или у нее нет ID
                     print(f"{Fore.YELLOW}Warning: Could not find previous task with ID to loop back to for task '{task['word']}'. Skipping loop creation.{Fore.RESET}")
                     # Добавляем исходную пару без изменений (или как-то иначе обработать?)
                     processed_pairs.append(pair)

            else:
                # Задача не в цикле, просто добавляем исходную пару
                processed_pairs.append(pair)
        else:
             # Если у пары уже нет ключа "task" (например, это уже обработанный цикл),
             # просто добавляем ее как есть.
             processed_pairs.append(pair)

    return processed_pairs


def find_previous_task(previous_tasks: list[dict], task: dict) -> dict | None:
    """
    Находит предыдущую задачу в списке предыдущих задач (у которых есть 'word'),
    которая имеет наибольшее семантическое сходство с текущей задачей ('word'),
    используя вызов DeepSeek (предполагается).
    Args:
        previous_tasks (list): list of previous tasks (словари с ключом 'word' и 'task_id')
        task (dict): the current task (словарь с ключом 'word')
    Returns:
        dict | None: the previous task dictionary with the highest similarity, or None
    """
    # Убедимся, что есть предыдущие задачи
    if not previous_tasks:
        return None

    # Формируем строку с предыдущими задачами для промпта
    # Важно: берем только задачи с task_id, чтобы было к чему возвращаться
    previous_tasks_with_ids = [t for t in previous_tasks if "task_id" in t]
    if not previous_tasks_with_ids:
        return None # Нет задач с ID, некуда возвращаться

    # Формируем строку вида "T0: task word 0\nT1: task word 1..."
    previous_tasks_str = "\n".join([f"{t['task_id']}: {t['word']}" for t in previous_tasks_with_ids])

    try:
        # Вызываем DeepSeek для определения, к какой предыдущей задаче относится текущая
        print(f"DEBUG: Finding previous task for '{task['word']}' among:\n{previous_tasks_str}")
        previous_task_text = prompts.find_previous_task(task["word"], previous_tasks_str)
        print(f"DEBUG: DeepSeek suggested loop back to: '{previous_task_text}'")

        # Ищем задачу с наибольшим сходством по тексту с ответом DeepSeek
        highest_similarity_task = None
        highest_similarity = -1 # Используем -1, чтобы любое совпадение было больше

        # Сначала ищем точное совпадение (если DeepSeek вернул точный текст задачи)
        exact_match = next((t for t in previous_tasks_with_ids if t['word'] == previous_task_text), None)
        if exact_match:
             highest_similarity_task = exact_match
             highest_similarity = 100 # Ставим максимальное сходство
        else:
            # Если точного совпадения нет, используем нечеткое сравнение (fuzz.ratio)
            # Сравниваем ответ DeepSeek с текстами предыдущих задач
            for prev_task in previous_tasks_with_ids:
                similarity = fuzz.ratio(previous_task_text, prev_task["word"])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    highest_similarity_task = prev_task

        # Добавим порог сходства, чтобы избежать неправильных связей
        similarity_threshold = 70 # Порог можно настроить
        if highest_similarity < similarity_threshold:
             print(f"{Fore.YELLOW}Warning: Low similarity ({highest_similarity}%) between DeepSeek suggestion '{previous_task_text}' and potential loop targets. Loop might be incorrect.{Fore.RESET}")
             # Можно вернуть None или все равно вернуть наилучшее найденное
             # return None # Раскомментировать, если нужна высокая уверенность

        return highest_similarity_task

    except Exception as e:
        print(f"{Fore.RED}Error during DeepSeek call in find_previous_task: {e}{Fore.RESET}")
        traceback.print_exc()
        # В случае ошибки можно попробовать просто вернуть последнюю задачу с ID
        return previous_tasks_with_ids[-1] if previous_tasks_with_ids else None


# --- Функции extract_exclusive_gateways, add_conditions, handle_text_with_conditions БЕЗ ИЗМЕНЕНИЙ ---
# Они работают либо с текстом, либо с парами, не трогая структуру агента.

def extract_exclusive_gateways(process_description: str, conditions: list) -> list:
    """
    Извлекает условия для каждого эксклюзивного шлюза из описания процесса
    Args:
        process_description (str): the process description
        conditions (list): the list of condition entities found in the process description
    Returns:
        list: the list of exclusive gateways
    """
    if not conditions: # Если условий нет, шлюзов тоже нет
        return []

    # Сортируем условия по началу, на всякий случай
    conditions.sort(key=lambda x: x['start'])

    first_condition_start = conditions[0]["start"]
    # Берем текст от первого условия до конца
    exclusive_gateway_text = process_description[first_condition_start:]

    response = "" # Инициализируем переменную
    try:
        if len(conditions) == 2:
            response = prompts.extract_exclusive_gateways_2_conditions(
                exclusive_gateway_text
            )
        elif len(conditions) > 2: # Используем общий промпт для >2 условий
             response = prompts.extract_exclusive_gateways(exclusive_gateway_text)
        else: # len(conditions) == 1 - одно условие не образует шлюз
            print(f"{Fore.YELLOW}Warning: Only one condition found, cannot form an exclusive gateway.{Fore.RESET}")
            return []
    except Exception as e:
        print(f"{Fore.RED}Error during DeepSeek call in extract_exclusive_gateways: {e}{Fore.RESET}")
        return [] # Возвращаем пустой список при ошибке

    # Парсим ответ DeepSeek для извлечения текстов шлюзов
    # Паттерн ищет "Exclusive gateway X: [текст шлюза]"
    pattern = r"Exclusive gateway \d+:\s*(.*?)(?=(?:Exclusive gateway \d+:|$))"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE) # Добавлен IGNORECASE
    gateway_texts = [s.strip() for s in matches if s.strip()] # Убираем пустые строки

    if not gateway_texts:
        print(f"{Fore.YELLOW}Warning: Could not parse exclusive gateways from DeepSeek response.{Fore.RESET}")
        print(f"DeepSeek response was: {response}")
        # Попытка: использовать исходные условия как один шлюз? (Не очень надежно)
        if len(conditions) >= 2:
             print(f"{Fore.YELLOW}Attempting to use original conditions as a single gateway.{Fore.RESET}")
             gateway_texts = [process_description[conditions[0]['start']:conditions[-1]['end']]]
        else:
             return []

    # Находим индексы текстов шлюзов в исходном описании
    try:
        gateway_indices = get_indices(gateway_texts, process_description)
        print("Exclusive gateway indices:", gateway_indices, "\n")
    except Exception as e:
         print(f"{Fore.RED}Error finding indices for gateway texts: {e}{Fore.RESET}")
         return []


    exclusive_gateways = []
    gateway_id_counter = 0

    # Связываем условия с найденными шлюзами
    # --- Упрощенная Логика: Используем промпт для группировки условий по шлюзам ---
    condition_string = "\n".join([f"- {c['word']} (start: {c['start']}, end: {c['end']})" for c in conditions])
    gateway_string = "\n".join([f"Gateway {i}: {text}" for i, text in enumerate(gateway_texts)])

    try:
        print("DEBUG: Grouping conditions by gateways using DeepSeek...")
        print(f"Conditions:\n{condition_string}")
        print(f"Gateways:\n{gateway_string}")
        grouping_response = prompts.extract_gateway_conditions(condition_string, gateway_string)
        print(f"DEBUG: DeepSeek grouping response:\n{grouping_response}")

        # Парсим ответ группировки (ожидаем формат типа "Gateway 0: condition1 || condition2 \n Gateway 1: condition3")
        gateway_conditions_map = {}
        pattern_group = r"Gateway (\d+):\s*(.*?)(?=(?:Gateway \d+:|$))"
        grouped_matches = re.findall(pattern_group, grouping_response, re.DOTALL | re.IGNORECASE)

        for gw_idx_str, conditions_str in grouped_matches:
            gw_idx = int(gw_idx_str)
            # Разделяем условия (могут быть разделены '||' или новой строкой)
            cond_list = [c.strip() for c in re.split(r'\s*\|\|\s*|\n', conditions_str) if c.strip()]
            if gw_idx < len(gateway_texts): # Убедимся, что индекс шлюза валидный
                gateway_conditions_map[gw_idx] = cond_list
            else:
                 print(f"{Fore.YELLOW}Warning: Parsed gateway index {gw_idx} out of range.{Fore.RESET}")

    except Exception as e:
         print(f"{Fore.RED}Error during DeepSeek call for grouping conditions: {e}{Fore.RESET}")
         # Откат: Попробуем присвоить условия ближайшему шлюзу (менее надежно)
         gateway_conditions_map = {} # Очищаем карту
         print(f"{Fore.YELLOW}Falling back to assigning conditions to nearest gateway.{Fore.RESET}")
         conditions_used = set()
         for gw_idx, gw_indices in enumerate(gateway_indices):
             gw_conditions = []
             for cond_idx, cond in enumerate(conditions):
                 if cond_idx not in conditions_used:
                     # Простое условие: условие находится внутри или близко к началу шлюза
                     if gw_indices['start'] <= cond['start'] < gw_indices['end']:
                         gw_conditions.append(cond['word'])
                         conditions_used.add(cond_idx)
             if gw_conditions:
                 gateway_conditions_map[gw_idx] = gw_conditions

    # Создаем финальную структуру шлюзов
    for gw_idx, gw_text in enumerate(gateway_texts):
        gw_id = f"EG{gateway_id_counter}"
        gateway_id_counter += 1

        conditions_for_this_gateway = gateway_conditions_map.get(gw_idx, [])
        if not conditions_for_this_gateway:
            print(f"{Fore.YELLOW}Warning: No conditions associated with Gateway {gw_idx} ('{gw_text[:50]}...'). Skipping gateway.{Fore.RESET}")
            continue

        # Находим индексы для условий этого шлюза
        try:
            condition_indices = get_indices(conditions_for_this_gateway, process_description)
        except Exception as e:
            print(f"{Fore.RED}Error finding indices for conditions of gateway {gw_idx}: {e}. Skipping gateway.{Fore.RESET}")
            continue

        exclusive_gateways.append({
            "id": gw_id,
            "conditions": conditions_for_this_gateway,
            "start": gateway_indices[gw_idx]["start"],
            "end": gateway_indices[gw_idx]["end"],
            "paths": condition_indices # Индексы путей = индексы условий
        })

    # --- Логика обработки вложенности и корректировки индексов путей (ОСТАВЛЕНА БЕЗ ИЗМЕНЕНИЙ) ---
    # Эта часть кода сложная и специфичная, требует внимательной проверки.
    # Check for nested exclusive gateways
    exclusive_gateways.sort(key=lambda x: x['start']) # Сортируем на всякий случай
    for i, gateway in enumerate(exclusive_gateways):
        if i + 1 < len(exclusive_gateways): # Проверяем следующий
            next_gateway = exclusive_gateways[i+1]
            # Условие вложенности: следующий шлюз начинается до конца текущего
            if next_gateway["start"] < gateway["end"]:
                 # Дополнительная проверка: конец следующего не выходит за конец текущего
                 # (или выходит не сильно далеко, чтобы избежать ложных срабатываний)
                 if next_gateway["end"] <= gateway["end"] + 10: # Добавим небольшой буфер
                     print(f"Nested exclusive gateway found: {next_gateway['id']} inside {gateway['id']}\n")
                     next_gateway["parent_gateway_id"] = gateway["id"]

    # Update the start and end indices of the paths
    for eg_idx, exclusive_gateway in enumerate(exclusive_gateways):
        exclusive_gateway['paths'].sort(key=lambda x: x['start']) # Сортируем пути
        num_paths = len(exclusive_gateway["paths"])
        for i, path in enumerate(exclusive_gateway["paths"]):
            # Конец текущего пути - это начало следующего пути (минус 1)
            if i < num_paths - 1:
                next_path_start = exclusive_gateway["paths"][i + 1]["start"]
                # Убедимся, что начало следующего пути больше конца текущего, чтобы избежать некорректных индексов
                if next_path_start > path['start']:
                    path["end"] = next_path_start - 1
                else:
                    # Если пути пересекаются или идут вплотную, ставим конец = начало + 1 (минимальная длина)
                    print(f"{Fore.YELLOW}Warning: Path overlap detected in EG {exclusive_gateway['id']}, path {i}. Adjusting end index.{Fore.RESET}")
                    path["end"] = path["start"] + 1 # Или использовать end из get_indices?

            # Обработка последнего пути шлюза
            else:
                # Если это вложенный шлюз
                if "parent_gateway_id" in exclusive_gateway:
                    parent_gateway = next((g for g in exclusive_gateways if g["id"] == exclusive_gateway["parent_gateway_id"]), None)
                    if parent_gateway:
                        # Ищем путь родителя, в который вложен текущий шлюз
                        parent_path_found = False
                        for parent_path in parent_gateway["paths"]:
                            if parent_path["start"] <= exclusive_gateway["start"] < parent_path["end"]:
                                # Конец последнего пути вложенного шлюза = конец пути родителя
                                path["end"] = parent_path["end"]
                                parent_path_found = True
                                break
                        if not parent_path_found:
                             print(f"{Fore.YELLOW}Warning: Could not find parent path for nested gateway {exclusive_gateway['id']}. Using gateway end.{Fore.RESET}")
                             path["end"] = exclusive_gateway["end"] # Откат
                    else: # Родитель не найден (ошибка в логике?)
                        print(f"{Fore.RED}Error: Parent gateway {exclusive_gateway['parent_gateway_id']} not found for nested gateway {exclusive_gateway['id']}. Using gateway end.{Fore.RESET}")
                        path["end"] = exclusive_gateway["end"] # Откат

                # Если это НЕ вложенный шлюз
                else:
                    # Ищем следующий НЕ ВЛОЖЕННЫЙ шлюз
                    next_outer_gateway = None
                    for next_gw_idx in range(eg_idx + 1, len(exclusive_gateways)):
                         if "parent_gateway_id" not in exclusive_gateways[next_gw_idx]:
                             next_outer_gateway = exclusive_gateways[next_gw_idx]
                             break

                    # Если есть следующий не вложенный шлюз, конец последнего пути = начало того шлюза
                    if next_outer_gateway:
                        if next_outer_gateway['start'] > path['start']: # Проверка корректности
                            path["end"] = next_outer_gateway["start"] - 1
                        else:
                             print(f"{Fore.YELLOW}Warning: Next outer gateway {next_outer_gateway['id']} starts before or at the end of path {i} in {exclusive_gateway['id']}. Using gateway end.{Fore.RESET}")
                             path["end"] = exclusive_gateway["end"] # Откат
                    # Если это последний шлюз в цепочке (или все последующие вложены)
                    else:
                         path["end"] = exclusive_gateway["end"] # Конец последнего пути = конец шлюза

    # Add parent gateway path id to the nested gateways
    for eg_idx, exclusive_gateway in enumerate(exclusive_gateways):
        if "parent_gateway_id" in exclusive_gateway:
            parent_gateway = next((g for g in exclusive_gateways if g["id"] == exclusive_gateway["parent_gateway_id"]), None)
            if parent_gateway:
                path_found = False
                for idx, parent_path in enumerate(parent_gateway["paths"]):
                    # Проверяем, что начало вложенного шлюза попадает в диапазон пути родителя
                    if parent_path["start"] <= exclusive_gateway["start"] < parent_path["end"]:
                        exclusive_gateway["parent_gateway_path_id"] = idx # Сохраняем индекс родительского пути
                        path_found = True
                        break
                if not path_found:
                     print(f"{Fore.YELLOW}Warning: Could not determine parent path ID for nested gateway {exclusive_gateway['id']}.{Fore.RESET}")
            # else: parent gateway not found error already printed

    print("Exclusive gateways data:", exclusive_gateways, "\n")
    return exclusive_gateways


def add_conditions(conditions: list, agent_task_pairs: list, sentences: list) -> list:
    """
    Adds conditions and condition ids to agent-task pairs.
    Args:
        conditions (list): a list of conditions (словари из NER)
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): a list of sentences data
    Returns:
        list: a list of agent-task pairs potentially with conditions added
    """
    condition_id_counter = 0
    assigned_conditions = set() # Чтобы не присваивать одно условие дважды

    for pair in agent_task_pairs:
        # Убедимся, что у пары есть задача и индекс предложения
        if "task" not in pair or "sentence_idx" not in pair:
            continue

        task = pair["task"]
        pair_sentence_idx = pair["sentence_idx"]

        # Ищем предложение, соответствующее паре
        current_sentence = None
        if 0 <= pair_sentence_idx < len(sentences):
             current_sentence = sentences[pair_sentence_idx]
             # Дополнительная проверка совпадения индексов на всякий случай
             if not (current_sentence['start'] <= task['start'] < current_sentence['end']):
                 # Попробуем найти правильное предложение по индексам задачи
                 current_sentence = next((s for s_idx, s in enumerate(sentences) if s['start'] <= task['start'] < s['end']), None)
                 if current_sentence:
                      pair['sentence_idx'] = sentences.index(current_sentence) # Обновляем индекс
                 else:
                      print(f"{Fore.YELLOW}Warning: Could not find sentence for task '{task['word']}' at index {pair_sentence_idx}. Cannot assign condition.{Fore.RESET}")
                      continue # Пропускаем пару, если не нашли предложение
        else:
             print(f"{Fore.YELLOW}Warning: Invalid sentence index {pair_sentence_idx} for task '{task['word']}'. Cannot assign condition.{Fore.RESET}")
             continue

        # Ищем условие в том же предложении
        for condition in conditions:
             condition_tuple = (condition['start'], condition['end']) # Уникальный идентификатор условия
             # Проверяем, что условие в том же предложении и еще не присвоено
             if current_sentence['start'] <= condition['start'] < current_sentence['end']:
                 if condition_tuple not in assigned_conditions:
                     # Присваиваем копию условия, чтобы избежать изменения оригинала при добавлении ID
                     condition_copy = condition.copy()
                     condition_copy["condition_id"] = f"C{condition_id_counter}"
                     pair["condition"] = condition_copy
                     assigned_conditions.add(condition_tuple)
                     condition_id_counter += 1
                     break # Нашли условие для этой пары, переходим к следующей паре

    return agent_task_pairs


def handle_text_with_conditions(
    agent_task_pairs: list, conditions: list, sents_data: list, process_desc: str
) -> tuple[list, list]: # Возвращает кортеж
    """
    Adds conditions to agent-task pairs and extracts exclusive gateway data.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        conditions (list): the list of condition entities
        sents_data (list): the sentence data
        process_desc (str): the process description
    Returns:
        tuple[list, list]: A tuple containing:
                           - the updated list of agent-task pairs with conditions
                           - the list of extracted exclusive gateways data
    """
    # Шаг 1: Добавить условия к парам агент-задача
    updated_agent_task_pairs = add_conditions(conditions, agent_task_pairs, sents_data)

    # Шаг 2: Извлечь данные об эксклюзивных шлюзах
    # Эта функция теперь зависит только от описания процесса и найденных условий
    exclusive_gateway_data = extract_exclusive_gateways(process_desc, conditions)

    return updated_agent_task_pairs, exclusive_gateway_data


def should_resolve_coreferences(text: str) -> bool:
    """
    Determines whether coreferences should potentially be resolved by checking for pronouns.
    Args:
        text (str): the text
    Returns:
        bool: True if coreferences should be resolved, False otherwise
    """
    # Используем простой набор местоимений
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "theirs"}
    # Быстрая проверка без spaCy для экономии времени
    # Разбиваем на слова и проверяем наличие местоимений в нижнем регистре
    words = re.findall(r'\b\w+\b', text.lower()) # Находим слова
    return any(word in pronouns for word in words)


def extract_all_entities(data: list | None, min_score: float) -> tuple:
    """
    Extracts all entities from the model output. Handles potential None input.
    Args:
        data (list | None): model output or None
        min_score (float): the minimum score for an entity to be extracted
    Returns:
        tuple: a tuple of lists containing the extracted entities (agents, tasks, conditions, process_info)
    """
    if data is None:
        print(f"{Fore.YELLOW}Warning: No data provided to extract_all_entities. Returning empty lists.{Fore.RESET}")
        return ([], [], [], []) # Возвращаем пустые списки

    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data, min_score)
    tasks = extract_entities("TASK", data, min_score)
    conditions = extract_entities("CONDITION", data, min_score)
    process_info = extract_entities("PROCESS_INFO", data, min_score)
    return (agents, tasks, conditions, process_info)


def get_indices(strings_to_find: list[str], text: str) -> list[dict]:
    """
    Gets the start and end indices of the given strings in the text by using fuzzy string matching.
    Handles cases where a string might appear multiple times, trying to find the best match.
    Args:
        strings_to_find (list): the list of strings to be found in the text
        text (str): the text in which the strings are to be found
    Returns:
        list[dict]: the list of dictionaries with 'start' and 'end' indices for each string.
                    Returns list of same length as strings_to_find.
    Raises:
        ValueError: If a string cannot be reasonably found in the text.
    """
    results = []
    text_lower = text.lower() # Для поиска без учета регистра

    for string_to_find in strings_to_find:
        if not string_to_find: # Пропускаем пустые строки
            results.append(None) # Добавляем None как маркер ошибки для этой строки
            continue

        string_lower = string_to_find.lower()
        best_match = None
        highest_similarity = -1
        search_start = 0

        # Ищем все возможные вхождения первого слова для начала поиска
        first_word = string_lower.split()[0]
        possible_starts = [m.start() for m in re.finditer(r'\b' + re.escape(first_word) + r'\b', text_lower)]

        if not possible_starts: # Если даже первое слово не найдено
             print(f"{Fore.YELLOW}Warning: First word '{first_word}' of '{string_to_find}' not found in text. Cannot find indices.{Fore.RESET}")
             # Попробуем найти строку целиком без привязки к первому слову
             try:
                  start_index = text_lower.index(string_lower)
                  end_index = start_index + len(string_to_find)
                  best_match = {"start": start_index, "end": end_index, "score": 100} # Считаем точным совпадением
                  highest_similarity = 100
             except ValueError:
                  results.append(None) # Строка не найдена вообще
                  continue # Переходим к следующей строке

        if not best_match: # Если не нашли точное совпадение на предыдущем шаге
            # Итеративно ищем лучшее совпадение по Fuzz ratio
            max_len = len(string_to_find) + 20 # Ищем в окне чуть большем длины строки

            for start_idx in possible_starts:
                substring_to_check = text[start_idx : min(start_idx + max_len, len(text))]
                # Используем fuzz.partial_ratio или ratio
                # ratio более строгий, partial_ratio лучше для подстрок
                similarity = fuzz.ratio(string_to_find, substring_to_check)

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    # Корректируем конец на основе найденного сходства (может быть неточно)
                    # Можно использовать find для уточнения, если сходство высокое
                    end_idx = start_idx + len(string_to_find) # Приблизительно
                    if similarity > 90: # Если сходство высокое, попробуем найти точный конец
                        try:
                             # Ищем исходную строку (с учетом регистра) рядом с найденным началом
                             actual_start = text.index(string_to_find, max(0, start_idx - 5), min(len(text), start_idx + 5))
                             end_idx = actual_start + len(string_to_find)
                             start_idx = actual_start
                        except ValueError:
                             pass # Оставляем приблизительный конец

                    best_match = {"start": start_idx, "end": end_idx, "score": similarity}

        similarity_threshold = 60 # Минимальное сходство для принятия результата
        if best_match and best_match['score'] >= similarity_threshold:
            results.append({"start": best_match["start"], "end": best_match["end"]})
        else:
            print(f"{Fore.YELLOW}Warning: Could not confidently find indices for '{string_to_find}'. Highest similarity: {highest_similarity}%.{Fore.RESET}")
            results.append(None) # Не удалось найти

    # Проверяем, что для всех строк найдены индексы
    if any(res is None for res in results):
        failed_strings = [s for s, res in zip(strings_to_find, results) if res is None]
        raise ValueError(f"Failed to find indices for the following strings: {failed_strings}")

    return results


# --- Функции get_parallel_paths, get_parallel_gateways, handle_text_with_parallel_keywords БЕЗ ИЗМЕНЕНИЙ ---
# Они работают с текстом и не зависят напрямую от структуры agent_task_pairs (кроме как для проверок)

def get_parallel_paths(parallel_gateway_text: str, process_description: str) -> list[dict] | None:
    """
    Возвращает начальный и конечный индексы параллельных путей в описании процесса.
    Args:
        parallel_gateway_text (str): the text identified as a parallel gateway by DeepSeek
        process_description (str): the full process description
    Returns:
        list[dict] | None: the list of start and end indices for each path, or None if cannot determine paths
    """
    try:
        print(f"DEBUG: Determining number of parallel paths for gateway text: '{parallel_gateway_text[:100]}...'")
        num_str = prompts.number_of_parallel_paths(parallel_gateway_text)
        num = int(num_str)
        print(f"DEBUG: DeepSeek suggests {num} parallel paths.")
    except ValueError:
        print(f"{Fore.YELLOW}Warning: Could not determine number of parallel paths from DeepSeek response '{num_str}'. Assuming 2 paths.{Fore.RESET}")
        num = 2 # Default or fallback
    except Exception as e:
        print(f"{Fore.RED}Error during DeepSeek call for number of parallel paths: {e}{Fore.RESET}")
        return None

    # Максимум 3 пути (ограничение промптов)
    if num > 3:
         print(f"{Fore.YELLOW}Warning: DeepSeek suggested {num} paths, but maximum supported is 3. Limiting to 3.{Fore.RESET}")
         num = 3

    if num <= 1:
        print("DEBUG: Only one or zero paths detected, no parallel execution needed for this gateway.")
        return None # Один путь - не параллельный шлюз

    paths_text = ""
    try:
        if num == 2:
            paths_text = prompts.extract_2_parallel_paths(parallel_gateway_text)
        elif num == 3:
            paths_text = prompts.extract_3_parallel_paths(parallel_gateway_text)
    except Exception as e:
         print(f"{Fore.RED}Error during DeepSeek call for extracting parallel paths: {e}{Fore.RESET}")
         return None

    # Разделяем пути (разделитель '&&' из промпта)
    paths = [s.strip() for s in paths_text.split("&&") if s.strip()]

    if len(paths) != num:
         print(f"{Fore.YELLOW}Warning: Expected {num} paths based on count, but parsed {len(paths)} from response '{paths_text}'. Using parsed paths.{Fore.RESET}")
         if not paths: return None # Не удалось распарсить пути

    # Находим индексы для каждого пути
    try:
        indices = get_indices(paths, process_description)
        print("Parallel path indices:", indices, "\n")
        return indices
    except ValueError as e:
        print(f"{Fore.RED}Error finding indices for parallel paths: {e}{Fore.RESET}")
        return None
    except Exception as e:
        print(f"{Fore.RED}Unexpected error finding indices for parallel paths: {e}{Fore.RESET}")
        return None


def get_parallel_gateways(text: str) -> list[dict]:
    """
    Получает индексы параллельных шлюзов (начало и конец области параллельного выполнения) в тексте.
    Args:
        text (str): the text
    Returns:
        list: the list of dictionaries with 'start' and 'end' indices for each gateway region.
    """
    try:
        print("DEBUG: Extracting parallel gateway regions using DeepSeek...")
        response = prompts.extract_parallel_gateways(text)
        print(f"DEBUG: DeepSeek response for parallel gateways:\n{response}")
    except Exception as e:
        print(f"{Fore.RED}Error during DeepSeek call in get_parallel_gateways: {e}{Fore.RESET}")
        return []

    # Паттерн для извлечения текста шлюза
    pattern = r"Parallel gateway \d+:\s*(.*?)(?=(?:Parallel gateway \d+:|$))"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    gateway_texts = [s.strip() for s in matches if s.strip()]

    if not gateway_texts:
         print(f"{Fore.YELLOW}Warning: Could not parse parallel gateways from DeepSeek response.{Fore.RESET}")
         return []

    # Находим индексы для текстов шлюзов
    try:
        indices = get_indices(gateway_texts, text)
        print("Parallel gateway region indices:", indices, "\n")
        return indices
    except ValueError as e:
         print(f"{Fore.RED}Error finding indices for parallel gateway texts: {e}{Fore.RESET}")
         return []
    except Exception as e:
         print(f"{Fore.RED}Unexpected error finding indices for parallel gateway texts: {e}{Fore.RESET}")
         return []


def handle_text_with_parallel_keywords(
    process_description: str, agent_task_pairs: list[dict], sents_data: list[dict]
) -> list[dict]:
    """
    Извлекает параллельные шлюзы и пути из описания процесса.
    Адаптирует agent_task_pairs для случая параллельных задач в одном предложении.
    Args:
        process_description (str): the process description
        agent_task_pairs (list): the list of agent-task pairs (может быть модифицирован!)
        sents_data (list): the sentence data
    Returns:
        list: the list of parallel gateways data structure.
    """
    parallel_gateways_data = []
    parallel_gateway_id_counter = 0

    # Получаем области текста, где DeepSeek нашел упоминания параллельности
    gateway_region_indices = get_parallel_gateways(process_description)

    for region_indices in gateway_region_indices:
        gateway_id = f"PG{parallel_gateway_id_counter}"
        parallel_gateway_id_counter += 1

        # Считаем задачи и предложения в этой области
        num_sentences_spanned = count_sentences_spanned(sents_data, region_indices, buffer=3)
        num_atp_in_region = num_of_agent_task_pairs_in_range(agent_task_pairs, region_indices)

        # --- Случай 1: Параллельность внутри одного предложения/одной задачи NER ---
        # (Модель NER может объединить несколько параллельных задач в одну сущность TASK)
        if num_sentences_spanned <= 1 and num_atp_in_region <= 1:
            print(f"Parallel gateway {gateway_id}: Handling potential parallel tasks within a single sentence/task entity.\n")

            sentence_text = get_sentence_text(sents_data, region_indices)
            if not sentence_text:
                 print(f"{Fore.YELLOW}Warning: Could not get sentence text for parallel region {gateway_id}. Skipping.{Fore.RESET}")
                 continue

            # Используем DeepSeek для явного извлечения параллельных задач из текста предложения/региона
            parallel_tasks_text = []
            try:
                 print(f"DEBUG: Extracting parallel tasks from: '{sentence_text}'")
                 response = prompts.extract_parallel_tasks(sentence_text)
                 parallel_tasks_text = extract_tasks(response) # Эта функция парсит "Task 1: ... Task 2: ..."
                 print(f"DEBUG: Extracted parallel tasks: {parallel_tasks_text}")
            except Exception as e:
                 print(f"{Fore.RED}Error during DeepSeek call for extracting parallel tasks: {e}{Fore.RESET}")
                 # Не удалось извлечь задачи, пропускаем этот шлюз или обрабатываем иначе?
                 continue # Пропускаем шлюз

            if not parallel_tasks_text or len(parallel_tasks_text) <= 1:
                 print(f"{Fore.YELLOW}Warning: Did not extract multiple parallel tasks for region {gateway_id}. Skipping parallel structure.{Fore.RESET}")
                 continue # Не нашли несколько задач, не создаем шлюз

            # Находим исходную пару агент-задача (если она была)
            original_atp_index = get_agent_task_pair_index(agent_task_pairs, region_indices)
            original_atp = None
            if original_atp_index is not None:
                 original_atp = agent_task_pairs[original_atp_index]
                 # Удаляем исходную пару, так как заменим ее новыми
                 agent_task_pairs.pop(original_atp_index)
                 print(f"DEBUG: Removed original ATP at index {original_atp_index} for parallel expansion.")
            else:
                 print(f"{Fore.YELLOW}Warning: No original agent-task pair found for parallel region {gateway_id}. Parallel tasks might lack an agent.{Fore.RESET}")
                 # Нужен агент по умолчанию? Или использовать предыдущего?
                 # Пока оставляем без агента или можно попробовать найти ближайшего предыдущего

            # Создаем новые пары агент-задача для каждой извлеченной параллельной задачи
            pg_path_indices = []
            insert_idx = original_atp_index if original_atp_index is not None else len(agent_task_pairs) # Куда вставлять новые пары
            task_start_offset = region_indices['start'] # Используем начало региона как базовую координату

            for i, task_word in enumerate(parallel_tasks_text):
                 # Создаем фиктивные координаты для новых задач внутри региона
                 # Важно: эти координаты могут не соответствовать точному тексту, они нужны для порядка
                 current_task_start = task_start_offset + i * 2 # Просто смещаем на 2 для разделения
                 current_task_end = current_task_start + 1

                 new_task_entity = {
                     "entity_group": "TASK",
                     "start": current_task_start,
                     "end": current_task_end,
                     "word": task_word,
                     "score": 0.99, # Ставим высокую уверенность, т.к. извлечено явно
                     "task_id": f"{gateway_id}_T{i}" # Уникальный ID для задачи внутри шлюза
                 }

                 # Создаем новую пару, копируя агента и sentence_idx из исходной (если была)
                 new_atp = {
                     "agent": original_atp['agent'] if original_atp else {"original_word": "Unknown", "resolved_word": "Unknown", "entity": None}, # Агент из исходной или Unknown
                     "task": new_task_entity,
                     "sentence_idx": original_atp['sentence_idx'] if original_atp else -1 # Индекс предложения или -1
                 }
                 agent_task_pairs.insert(insert_idx + i, new_atp) # Вставляем новые пары

                 pg_path_indices.append({
                     "start": current_task_start,
                     "end": current_task_end,
                     "task_id_ref": new_task_entity["task_id"] # Ссылка на ID созданной задачи
                 })

            # Создаем структуру данных для параллельного шлюза
            gateway_data = {
                "id": gateway_id,
                "start": pg_path_indices[0]["start"],
                "end": pg_path_indices[-1]["end"],
                "paths": pg_path_indices,
                "type": "single_sentence_expansion" # Маркер типа шлюза
            }
            parallel_gateways_data.append(gateway_data)

        # --- Случай 2: Параллельность охватывает несколько предложений/задач ---
        else:
            print(f"Parallel gateway {gateway_id}: Handling parallel execution across multiple sentences/tasks.\n")
            gateway_text = process_description[region_indices["start"] : region_indices["end"]]
            # Получаем индексы для каждого параллельного пути
            path_indices = get_parallel_paths(gateway_text, process_description)

            if path_indices: # Если удалось получить пути
                gateway_data = {
                    "id": gateway_id,
                    "start": region_indices["start"],
                    "end": region_indices["end"],
                    "paths": path_indices, # Используем индексы, найденные get_parallel_paths
                    "type": "multi_sentence_region"
                }
                parallel_gateways_data.append(gateway_data)
            else:
                 print(f"{Fore.YELLOW}Warning: Could not determine parallel paths for gateway region {gateway_id}. Skipping parallel structure.{Fore.RESET}")


    # --- Обработка вложенных параллельных шлюзов (ОСТАВЛЕНА БЕЗ ИЗМЕНЕНИЙ) ---
    # Эта логика сложна и зависит от точности get_parallel_paths
    parallel_gateways_data.sort(key=lambda x: x['start']) # Сортируем перед проверкой вложенности
    gateways_to_add = [] # Буфер для новых вложенных шлюзов

    for gateway in parallel_gateways_data:
        # Пропускаем уже созданные вложенные шлюзы и шлюзы типа single_sentence
        if "parallel_parent" in gateway or gateway.get("type") == "single_sentence_expansion":
            continue

        if "paths" in gateway and gateway["paths"]: # Убедимся, что пути существуют
            for path_idx, path in enumerate(gateway["paths"]):
                 path_text = process_description[path["start"] : path["end"]]
                 # Проверяем, есть ли ключевые слова параллельности *внутри* этого пути
                 if has_parallel_keywords(path_text):
                     print(f"Nested parallel keywords detected in path {path_idx} of gateway {gateway['id']}: '{path_text[:50]}...'\n")
                     # Находим индексы для вложенных путей
                     nested_path_indices = get_parallel_paths(path_text, process_description)
                     if nested_path_indices:
                         nested_gateway_id = f"PG{parallel_gateway_id_counter}"
                         parallel_gateway_id_counter += 1
                         nested_gateway_data = {
                             "id": nested_gateway_id,
                             # Начало/конец вложенного шлюза = начало/конец первого/последнего вложенного пути
                             "start": nested_path_indices[0]["start"],
                             "end": nested_path_indices[-1]["end"],
                             "paths": nested_path_indices,
                             "parallel_parent": gateway["id"], # Ссылка на родительский шлюз
                             "parallel_parent_path_idx": path_idx, # Индекс пути родителя
                             "type": "nested_region"
                         }
                         gateways_to_add.append(nested_gateway_data)
                     else:
                          print(f"{Fore.YELLOW}Warning: Could not determine nested parallel paths for text in path {path_idx} of {gateway['id']}.{Fore.RESET}")

    parallel_gateways_data.extend(gateways_to_add) # Добавляем найденные вложенные шлюзы
    parallel_gateways_data.sort(key=lambda x: x['start']) # Сортируем снова

    print("Parallel gateway data:", parallel_gateways_data, "\n")
    return parallel_gateways_data


# --- Функции num_of_agent_task_pairs_in_range, get_agent_task_pair_index, count_sentences_spanned, get_sentence_text, extract_tasks БЕЗ ИЗМЕНЕНИЙ ---

def num_of_agent_task_pairs_in_range(
    agent_task_pairs: list[dict], indices: dict[str, int]
) -> int:
    """
    Возвращает количество пар агент-задача, в которых задача попадает в заданный диапазон индексов.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        indices (dict): the start and end indices {'start': N, 'end': M}
    Returns:
        int: the number of agent-task pairs in the given range
    """
    count = 0
    start_idx = indices.get('start', -1)
    end_idx = indices.get('end', -1)

    if start_idx == -1 or end_idx == -1:
        return 0 # Некорректные индексы

    for pair in agent_task_pairs:
        # Проверяем наличие задачи и ее индексов
        if "task" in pair and isinstance(pair["task"], dict):
            task_start = pair["task"].get("start", -1)
            task_end = pair["task"].get("end", -1)
            if task_start != -1 and task_end != -1:
                 # Задача должна полностью или частично попадать в диапазон
                 # Условие: (начало_задачи < конец_диапазона) И (конец_задачи > начало_диапазона)
                 if task_start < end_idx and task_end > start_idx:
                     count += 1
    return count


def get_agent_task_pair_index(
    agent_task_pairs: list[dict], indices: dict[str, int]
) -> int | None: # Возвращает индекс или None
    """
    Возвращает индекс первой пары агент-задача, задача которой попадает в заданный диапазон индексов.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        indices (dict): the start and end indices {'start': N, 'end': M}
    Returns:
        int | None: the index of the agent-task pair, or None if not found
    """
    start_idx = indices.get('start', -1)
    end_idx = indices.get('end', -1)

    if start_idx == -1 or end_idx == -1:
        return None

    for idx, pair in enumerate(agent_task_pairs):
        if "task" in pair and isinstance(pair["task"], dict):
            task_start = pair["task"].get("start", -1)
            task_end = pair["task"].get("end", -1)
            if task_start != -1 and task_end != -1:
                if task_start < end_idx and task_end > start_idx:
                    return idx # Возвращаем индекс первой найденной пары
    return None # Не найдено


def count_sentences_spanned(
    sentence_data: list[dict], indices: dict[str, int], buffer: int = 0 # Убрал буфер по умолчанию
) -> int:
    """
    Подсчитывает количество предложений, охватываемых заданным диапазоном индексов.
    Args:
        sentence_data (list): the list of sentence data
        indices (dict): the start and end indices {'start': N, 'end': M}
        buffer (int): buffer to potentially ignore start/end (usually 0)
    Returns:
        int: the number of sentences spanned by the given range
    """
    count = 0
    idx_start = indices.get('start', -1) + buffer
    idx_end = indices.get('end', -1) - buffer

    if idx_start == -1 or idx_end == -1 or idx_start >= idx_end:
        return 0

    for sentence_info in sentence_data:
        sent_start = sentence_info.get('start', -1)
        sent_end = sentence_info.get('end', -1)
        if sent_start != -1 and sent_end != -1:
            # Предложение пересекается с диапазоном, если:
            # (начало_предложения < конец_диапазона) И (конец_предложения > начало_диапазона)
            if sent_start < idx_end and sent_end > idx_start:
                count += 1
    return count


def get_sentence_text(sentence_data: list[dict], indices: dict[str, int]) -> str | None:
    """
    Возвращает текст первого предложения, который совпадает с заданным диапазоном индексов.
    Args:
        sentence_data (list): the list of sentence data
        indices (dict): the start and end indices {'start': N, 'end': M}
    Returns:
        str | None: the text of the sentence, or None if no overlapping sentence is found
    """
    idx_start = indices.get('start', -1)
    idx_end = indices.get('end', -1)

    if idx_start == -1 or idx_end == -1:
        return None

    for sentence_info in sentence_data:
        sent_start = sentence_info.get('start', -1)
        sent_end = sentence_info.get('end', -1)
        if sent_start != -1 and sent_end != -1:
             if sent_start < idx_end and sent_end > idx_start:
                 return sentence_info.get('sentence') # Возвращаем текст первого найденного предложения
    return None # Не найдено пересекающихся предложений


def extract_tasks(model_response: str) -> list[str]:
    """
    Обрабатывает ответ модели (ожидаемый "Задача 1: ... Задача 2: ...") для извлечения описаний задач.
    Args:
        model_response (str): the model response from prompts.extract_parallel_tasks
    Returns:
        list: the list of task description strings
    """
    if not model_response:
         return []
    # Паттерн ищет "Task X: [текст задачи]" до следующего "Task" или конца строки
    pattern = r"Task \d+:\s*(.*?)(?=(?:Task \d+:|$))"
    matches = re.findall(pattern, model_response, re.DOTALL | re.IGNORECASE) # Добавлен IGNORECASE
    tasks = [s.strip() for s in matches if s.strip()]
    return tasks


# --- Основная функция process_text С ИЗМЕНЕНИЯМИ КОРЕФЕРЕНЦИИ ---
def process_text(text: str) -> list[dict] | None: # Уточнили возвращаемый тип
    """
    Обрабатывает текст для создания структуры BPMN (списка словарей).
    Args:
        text (str): the text of the process description
    Returns:
        list[dict] | None: the list of dictionaries representing the BPMN structure, or None on failure
    """
    clear_folder("./output_logs")
    print(f"\n{Fore.CYAN}--- Starting BPMN Processing ---{Fore.RESET}")
    print(f"\nInput text:\n{text}\n")

    # --- БЛОК КОРЕФЕРЕНЦИИ (ИЗМЕНЕН) ---
    coref_info = None
    original_text = text # Сохраняем исходный текст
    clusters = None
    # Проверяем, нужно ли запускать и доступна ли модель
    if coref_model and should_resolve_coreferences(original_text):
        print("Attempting coreference resolution...\n")
        coref_info = get_coref_info(original_text, print_clusters=True) # Включаем печать кластеров для отладки
        if coref_info:
            clusters = coref_info.get('clusters_str') # Получаем кластеры строк
            print(f"{Fore.GREEN}Coreference resolution successful. Found {len(clusters) if clusters else 0} clusters.{Fore.RESET}\n")
            # original_text уже есть в coref_info['text'], можно использовать его для консистентности
            # original_text = coref_info['text'] # Раскомментировать, если текст из coref_info может отличаться
            write_to_file("coref_clusters.json", clusters if clusters else [])
        else:
            # Ошибка внутри get_coref_info (уже напечатана)
            print(f"{Fore.YELLOW}Coreference resolution failed or returned no info. Proceeding without coreference data.{Fore.RESET}\n")
            # clusters останется None
    else:
        if not coref_model:
             print(f"{Fore.YELLOW}Coref model not loaded. Skipping coreference resolution.{Fore.RESET}\n")
        else:
             print("No relevant pronouns found, skipping coreference resolution.\n")
    # --- КОНЕЦ БЛОКА КОРЕФЕРЕНЦИИ ---

    # Шаг 1: Извлечение сущностей NER
    # Используем original_text
    bpmn_entities = extract_bpmn_data(original_text)
    if bpmn_entities is None:
        print(f"{Fore.RED}Failed to extract BPMN entities. Aborting process.{Fore.RESET}")
        return None # Критическая ошибка, прерываем выполнение

    # Шаг 2: Исправление разбитых токенов (если нужно)
    bpmn_entities = fix_bpmn_data(bpmn_entities)
    print("Initial BPMN Entities (after fixing):")
    print(json.dumps(bpmn_entities, indent=2)) # Печатаем результат NER

    # Шаг 3: Разделение на типы сущностей
    # Используем результат шага 2
    agents, tasks, conditions, process_info = extract_all_entities(bpmn_entities, min_score=0.7) # Повысим порог?
    print(f"Extracted: {len(agents)} Agents, {len(tasks)} Tasks, {len(conditions)} Conditions, {len(process_info)} Process Infos\n")
    parallel_gateway_data = []
    exclusive_gateway_data = []

    # Шаг 4: Получение данных о предложениях
    # Используем original_text
    sents_data = create_sentence_data(original_text)

    # Шаг 5: Создание пар Агент-Задача с учетом кореференции
    # Передаем кластеры, полученные ранее
    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data, clusters)
    print(f"Initial Agent-Task Pairs ({len(agent_task_pairs)}):")
    # Выведем несколько пар для проверки разрешенных агентов
    for i, pair in enumerate(agent_task_pairs[:5]):
         agent_info = pair.get('agent', {})
         task_info = pair.get('task', {})
         print(f"  Pair {i}: Agent='{agent_info.get('resolved_word', 'N/A')}' (Orig='{agent_info.get('original_word', 'N/A')}'), Task='{task_info.get('word', 'N/A')}'")
    print("...\n")
    write_to_file("agent_task_pairs_initial.json", agent_task_pairs)

    # Шаг 6: Обработка параллельных шлюзов (если есть ключевые слова)
    # Используем original_text для поиска ключевых слов и анализа
    if has_parallel_keywords(original_text):
        print("Parallel keywords detected. Handling parallel gateways...\n")
        parallel_gateway_data = handle_text_with_parallel_keywords(
            original_text, agent_task_pairs, sents_data # agent_task_pairs может быть изменен внутри!
        )
        write_to_file("parallel_gateway_data.json", parallel_gateway_data)
    else:
        print("No parallel keywords detected.\n")

    # Шаг 7: Обработка эксклюзивных шлюзов (если есть условия)
    if len(conditions) > 0:
        print("Conditions detected. Handling exclusive gateways...\n")
        # Используем original_text для анализа
        # Эта функция также добавляет 'condition' к agent_task_pairs
        agent_task_pairs, exclusive_gateway_data = handle_text_with_conditions(
            agent_task_pairs, conditions, sents_data, original_text
        )
        write_to_file("exclusive_gateway_data.json", exclusive_gateway_data)
    else:
        print("No conditions detected.\n")

    # Шаг 8: Классификация Process Info и добавление конечных событий
    if len(process_info) > 0:
        print("Handling PROCESS_INFO entities...\n")
        process_info = batch_classify_process_info(process_info)
        # Эта функция добавляет 'process_end_event' к agent_task_pairs
        agent_task_pairs = add_process_end_events(
            agent_task_pairs, sents_data, process_info
        )
        write_to_file("process_info_entities_classified.json", process_info)
    else:
        print("No PROCESS_INFO entities found.\n")


    # Шаг 9: Обработка циклов (Loop)
    print("Handling loops...\n")
    loop_sentences = find_sentences_with_loop_keywords(sents_data)
    if loop_sentences:
         print(f"Found {len(loop_sentences)} potential loop sentences.")
    # Сначала добавляем ID ко всем задачам, *не* в цикле
    agent_task_pairs = add_task_ids(agent_task_pairs, sents_data, loop_sentences)
    # Затем преобразуем элементы цикла, добавляя 'go_to'
    agent_task_pairs = add_loops(agent_task_pairs, sents_data, loop_sentences)


    print("Final Agent-Task Pairs / Loop Elements:")
    print(json.dumps(agent_task_pairs, indent=2)) # Печатаем финальную структуру пар
    write_to_file("agent_task_pairs_final.json", agent_task_pairs)

    # Шаг 10: Создание финальной структуры BPMN
    # ВАЖНО: Убедитесь, что create_bpmn_structure корректно использует новую структуру
    # agent_task_pairs, особенно pair['agent']['resolved_word'] для дорожек (lanes).
    print("\nCreating final BPMN structure...\n")
    final_structure = create_bpmn_structure(
        agent_task_pairs, parallel_gateway_data, exclusive_gateway_data, process_info
    )

    if final_structure is None:
         print(f"{Fore.RED}Failed to create BPMN structure.{Fore.RESET}")
         return None

    print("BPMN Structure created successfully.")
    write_to_file("bpmn_structure.json", final_structure)

    print(f"\n{Fore.CYAN}--- BPMN Processing Finished ---{Fore.RESET}")
    return final_structure


# --- Функции generate_graph_pdf и generate_graph_image БЕЗ ИЗМЕНЕНИЙ ---
def generate_graph_pdf(input: list[dict], notebook: bool) -> None:
    """
    Создает график структуры BPMN и сохраняет его в формате PDF. РАБОТАЕТ КРИВО!!! Первично смотреть bpmn_final_structure.json
    Args:
        input (list): the list of dictionaries representing the BPMN structure
        notebook (bool): whether the graph is being generated in a Jupyter notebook
    """
    if not input:
        print(f"{Fore.YELLOW}Warning: No BPMN structure provided to generate graph.{Fore.RESET}")
        return
    bpmn = GraphGenerator(input, notebook=notebook)
    print("Generating graph (PDF)...\n")
    bpmn.generate_graph()
    bpmn.show()


def generate_graph_image(input: list[dict]) -> None:
    """
    Создает график структуры BPMN и сохраняет его в виде изображения. РАБОТАЕТ КРИВО!!! Первично смотреть bpmn_final_structure.json
    Args:
        input (list): the list of dictionaries representing the BPMN structure
    """
    if not input:
        print(f"{Fore.YELLOW}Warning: No BPMN structure provided to generate graph.{Fore.RESET}")
        return
    bpmn = GraphGenerator(input, format="jpeg", notebook=False)
    print("Generating graph (JPEG)...\n")
    bpmn.generate_graph()
    bpmn.save_file()

# --- END OF FILE process_bpmn_data.py ---