# --- START OF FILE process_bpmn_data.py ---

import json
import re
import traceback
import os
import requests
import spacy
from colorama import Fore, init # Добавлен init для Windows
from spacy.matcher import Matcher
from thefuzz import fuzz, process
import deepseek_prompts as prompts
from coreference_resolution.coref import get_coref_info, coref_model
from create_bpmn_structure import create_bpmn_structure
from logging_utils import clear_folder, write_to_file
from dotenv import load_dotenv
from collections import defaultdict # <--- ДОБАВЛЕНО для новой логики

# Инициализация colorama (особенно для Windows)
init(autoreset=True)

# --- Константы и загрузка моделей (без изменений) ---
BPMN_INFORMATION_EXTRACTION_ENDPOINT = "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction-v2"
ZERO_SHOT_CLASSIFICATION_ENDPOINT = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)
load_dotenv()
HF_API_TOKEN = os.getenv("HF_TOKEN")

# --- spaCy модели ---
try:
    nlp_sm = spacy.load("en_core_web_sm")
    nlp_md = spacy.load("en_core_web_md")
except OSError as e:
    print(f"{Fore.RED}Error loading spaCy models: {e}{Fore.RESET}")
    print(f"{Fore.YELLOW}Please run 'python -m spacy download en_core_web_sm' and 'python -m spacy download en_core_web_md'{Fore.RESET}")
    nlp_sm, nlp_md = None, None # Устанавливаем в None, чтобы проверки ниже работали

# --- Функции get_sentences до _resolve_agent_mention (без изменений) ---
def get_sentences(text: str) -> list[str]:
    """
    Creates a list of sentences from a given text using the preloaded spaCy model.
    """
    if not nlp_sm: # Проверка, загрузилась ли модель
        print(f"{Fore.RED}spaCy 'en_core_web_sm' model not loaded. Cannot split sentences.{Fore.RESET}")
        # Пытаемся разделить по точкам как fallback
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences: return [text] # Если не удалось разделить, возвращаем как есть
        # Добавляем точку обратно, если она была не в конце
        sentences = [s + '.' if not text.endswith(s) else s for s in sentences]
        return sentences
    doc = nlp_sm(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()] # Используем sent.text и убираем пустые
    return sentences

def create_sentence_data(text: str) -> list[dict]:
    """
    Создает список словарей, содержащих данные предложения (предложение, начальный индекс, конечный индекс).
    """
    sentences = get_sentences(text)
    sentence_data = []
    current_pos = 0

    # --- ДОБАВЛЕНА глобальная переменная для использования в extract_exclusive_gateways ---
    global sents_data # Объявляем, что будем использовать глобальную переменную
    sents_data = [] # Инициализируем или очищаем перед заполнением
    # --- КОНЕЦ ДОБАВЛЕНИЯ ---


    for sent_text in sentences:
        try:
            # Используем find для более надежного поиска, игнорируя регистр для начала
            start = text.find(sent_text, current_pos)
            if start == -1: # Если точное совпадение не найдено, попробуем без учета регистра
                 start = text.lower().find(sent_text.lower(), current_pos)

            if start != -1:
                end = start + len(sent_text)
                data_item = {"sentence": sent_text, "start": start, "end": end} # Создаем элемент
                sentence_data.append(data_item)
                sents_data.append(data_item) # Добавляем в глобальную переменную
                current_pos = end
            else: # Если найти не удалось совсем
                 print(f"{Fore.YELLOW}Warning: Could not accurately find sentence boundaries for: '{sent_text}'. Indices might be missing.{Fore.RESET}")
        except Exception as e:
            print(f"{Fore.RED}Error finding sentence boundaries for '{sent_text}': {e}{Fore.RESET}")

    return sentence_data

def query(payload: dict, endpoint: str) -> dict:
    """
    Отправляет POST-запрос на указанную конечную точку.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = json.dumps(payload)
    try:
        response = requests.post(endpoint, data=data, headers=headers, timeout=45) # Увеличен таймаут
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"{Fore.RED}Request timed out after 45 seconds for endpoint {endpoint}.{Fore.RESET}")
        return {"error": "Request timed out", "status_code": 408}
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}HTTP Request failed: {e}{Fore.RESET}")
        status_code = getattr(e.response, 'status_code', None)
        error_detail = getattr(e.response, 'text', str(e))
        print(f"{Fore.YELLOW}Status Code: {status_code}, Detail: {error_detail[:500]}{Fore.RESET}")
        return {"error": f"HTTP Request failed: {error_detail}", "status_code": status_code}
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}Failed to decode JSON response: {e}{Fore.RESET}")
        raw_response = getattr(response, 'text', 'N/A')
        print(f"{Fore.YELLOW}Raw response (partial): {raw_response[:500]}{Fore.RESET}")
        return {"error": f"Failed to decode JSON: {str(e)}", "raw_response": raw_response}
    except Exception as e:
        print(f"{Fore.RED}An unexpected error occurred during API query: {e}{Fore.RESET}")
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}

def extract_bpmn_data(text: str) -> list[dict] | None:
    """
    Извлекает данные BPMN из описания процесса.
    """
    print("Extracting BPMN data...\n")
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    data = query(payload, BPMN_INFORMATION_EXTRACTION_ENDPOINT)

    if isinstance(data, dict) and "error" in data:
        print(f"{Fore.RED}Error when extracting BPMN data: {data['error']}{Fore.RESET}")
        if "status_code" in data:
             print(f"Status Code: {data['status_code']}")
        return None
    elif not isinstance(data, list):
        print(f"{Fore.RED}Unexpected response format from BPMN extraction model. Expected list, got {type(data)}.{Fore.RESET}")
        print(f"Response: {data}")
        return None

    return data

def fix_bpmn_data(data: list[dict]) -> list[dict]:
    """
    Исправляет вывод модели NER, объединяя разбитые токены задач.
    """
    if not data: return []
    fixed_data = []
    i = 0
    while i < len(data):
        current_entity = data[i]
        if i + 1 < len(data):
            next_entity = data[i+1]
            if (current_entity["entity_group"] == "TASK"
                and next_entity["entity_group"] == "TASK"
                and current_entity["end"] == next_entity["start"]):
                combined_word = current_entity["word"]
                next_word_part = next_entity["word"]
                if next_word_part.startswith("##"): combined_word += next_word_part[2:]
                elif combined_word[-1].isalnum() and next_word_part[0].isalnum(): combined_word += next_word_part
                else: combined_word += next_word_part
                current_entity["word"] = combined_word
                current_entity["end"] = next_entity["end"]
                current_entity["score"] = max(current_entity["score"], next_entity["score"])
                i += 1; continue
        fixed_data.append(current_entity)
        i += 1
    if len(fixed_data) != len(data): print("BPMN data fixed (combined TASK tokens).")
    return fixed_data

def classify_process_info(text: str) -> dict | None:
    """
    Классифицирует объект PROCESS_INFO.
    """
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": ["beginning", "end", "split", "going back", "continuation"]},
        "options": {"wait_for_model": True},
    }
    data = query(payload, ZERO_SHOT_CLASSIFICATION_ENDPOINT)

    if isinstance(data, dict) and "error" in data:
        print(f"{Fore.RED}Error when classifying PROCESS_INFO entity '{text}': {data['error']}{Fore.RESET}")
        return None
    elif not isinstance(data, dict) or not all(k in data for k in ["sequence", "labels", "scores"]):
         print(f"{Fore.RED}Unexpected response format from Zero-Shot model for '{text}'.{Fore.RESET}")
         print(f"Response: {data}")
         return None
    return data

def batch_classify_process_info(process_info_entities: list[dict]) -> list[dict]:
    """
    Классифицирует список PROCESS_INFO сущностей.
    """
    updated_entities = []
    print("Classifying PROCESS_INFO entities...\n")
    process_info_map = {"beginning": "PROCESS_START", "end": "PROCESS_END", "split": "PROCESS_SPLIT", "going back": "PROCESS_RETURN", "continuation": "PROCESS_CONTINUE"}
    for entity in process_info_entities:
        text = entity["word"]
        classification_result = classify_process_info(text)
        if classification_result:
            top_label = classification_result["labels"][0]
            entity["entity_group"] = process_info_map.get(top_label, "PROCESS_INFO")
            entity["classification_score"] = classification_result["scores"][0]
        else:
            print(f"{Fore.YELLOW}Could not classify '{text}', leaving as PROCESS_INFO.{Fore.RESET}")
            entity["entity_group"] = "PROCESS_INFO"
        updated_entities.append(entity)
    return updated_entities

def extract_entities(type: str, data: list[dict], min_score: float) -> list[dict]:
    """
    Извлекает все сущности заданного типа из вывода модели NER.
    """
    if not data: return []
    return [
        entity for entity in data
        if entity.get("entity_group") == type and entity.get("score", 0) >= min_score
    ]

def _resolve_agent_mention(agent_entity: dict, clusters: list[list[str]] | None) -> tuple[str, bool]:
    """
    Находит репрезентативное упоминание для агента, используя кластеры кореференций.
    """
    if not agent_entity or 'word' not in agent_entity: return "", False
    agent_word = agent_entity['word']
    resolved_name = agent_word
    is_resolved = False
    if clusters:
        agent_word_lower = agent_word.lower()
        pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "theirs"}
        for cluster in clusters:
            cluster_lower = [c.lower() for c in cluster]
            if agent_word_lower in cluster_lower:
                representative_mention = cluster[0]
                first_non_pronoun = next((mention for mention in cluster if mention.lower() not in pronouns), None)
                if first_non_pronoun: representative_mention = first_non_pronoun
                rep_lower = representative_mention.lower()
                should_resolve = agent_word_lower in pronouns or (agent_word_lower != rep_lower)
                if agent_word != representative_mention and should_resolve:
                     resolved_name = representative_mention; is_resolved = True
                break
    return resolved_name, is_resolved

def create_agent_task_pairs(
    agents: list[dict], tasks: list[dict], sentence_data: list[dict], clusters: list[list[str]] | None = None
) -> list[dict]:
    """
    Объединяет агентов и задачи в пары, разрешая кореференции агентов.
    """
    agents_in_sentences = [{"sentence_idx": i, "agent": agent} for agent in agents for i, sent in enumerate(sentence_data) if sent["start"] <= agent["start"] < sent["end"]]
    tasks_in_sentences = [{"sentence_idx": i, "task": task} for task in tasks for i, sent in enumerate(sentence_data) if sent["start"] <= task["start"] < sent["end"]]

    multi_agent_sentences_idx = set()
    agent_counts_per_sentence = defaultdict(int)
    for agent_sent in agents_in_sentences:
        idx = agent_sent["sentence_idx"]
        agent_counts_per_sentence[idx] += 1
        if agent_counts_per_sentence[idx] > 1: multi_agent_sentences_idx.add(idx)

    agent_task_pairs = []
    processed_tasks_indices = set()

    single_agent_tasks = defaultdict(list); single_agent_info = {}
    for agent_sent in agents_in_sentences:
        sent_idx = agent_sent["sentence_idx"]
        if sent_idx in multi_agent_sentences_idx: continue
        agent_entity = agent_sent['agent']; resolved_name, _ = _resolve_agent_mention(agent_entity, clusters)
        if sent_idx not in single_agent_info: single_agent_info[sent_idx] = {"original_word": agent_entity['word'], "resolved_word": resolved_name, "entity": agent_entity}
        tasks_for_this_agent = [task_sent['task'] for task_sent in tasks_in_sentences if task_sent['sentence_idx'] == sent_idx]
        for task_entity in tasks_for_this_agent:
            task_tuple = (task_entity['start'], task_entity['end'])
            if task_tuple not in processed_tasks_indices:
                single_agent_tasks[sent_idx].append(task_entity); processed_tasks_indices.add(task_tuple)

    for sent_idx, tasks_list in single_agent_tasks.items():
         agent_info = single_agent_info.get(sent_idx)
         if agent_info:
             for task_entity in tasks_list:
                  pair = {"agent": agent_info, "task": task_entity, "sentence_idx": sent_idx}; agent_task_pairs.append(pair)

    if multi_agent_sentences_idx:
        multi_agent_task_pairs = handle_multi_agent_sentences(agents_in_sentences, tasks_in_sentences, list(multi_agent_sentences_idx), clusters)
        for pair in multi_agent_task_pairs:
             task_tuple = (pair['task']['start'], pair['task']['end'])
             if task_tuple not in processed_tasks_indices:
                 agent_task_pairs.append(pair); processed_tasks_indices.add(task_tuple)

    agent_task_pairs.sort(key=lambda k: (k.get("sentence_idx", -1), k.get("task", {}).get("start", float('inf'))))
    return agent_task_pairs

def handle_multi_agent_sentences(
    agents_in_sentences: list[dict], tasks_in_sentences: list[dict], multi_agent_sentences_idx: list[int], clusters: list[list[str]] | None = None
) -> list[dict]:
    """
    Создает пары агент-задача для предложений с несколькими агентами (эвристика).
    """
    agent_task_pairs = []
    for idx in multi_agent_sentences_idx:
        agents_in_this_sentence = sorted([agent_sent['agent'] for agent_sent in agents_in_sentences if agent_sent['sentence_idx'] == idx], key=lambda x: x['start'])
        tasks_in_this_sentence = sorted([task_sent['task'] for task_sent in tasks_in_sentences if task_sent['sentence_idx'] == idx], key=lambda x: x['start'])
        for task_entity in tasks_in_this_sentence:
            closest_preceding_agent = None; min_distance = float('inf')
            for agent_entity in agents_in_this_sentence:
                if agent_entity['end'] <= task_entity['start']:
                    distance = task_entity['start'] - agent_entity['end']
                    if distance < min_distance: min_distance = distance; closest_preceding_agent = agent_entity
            if closest_preceding_agent:
                resolved_name, _ = _resolve_agent_mention(closest_preceding_agent, clusters)
                pair = {"agent": {"original_word": closest_preceding_agent['word'], "resolved_word": resolved_name, "entity": closest_preceding_agent}, "task": task_entity, "sentence_idx": idx}
                agent_task_pairs.append(pair)
            else: print(f"{Fore.YELLOW}Warning: Could not associate a *preceding* agent for task '{task_entity['word']}' in multi-agent sentence {idx}. Skipping task pairing.{Fore.RESET}")
    return agent_task_pairs

def add_process_end_events(agent_task_pairs: list[dict], sentences: list[dict], process_info_entities: list[dict]) -> list[dict]:
    process_end_events_map = {}
    for entity in process_info_entities:
        if entity.get("entity_group") == "PROCESS_END":
            for i, sent in enumerate(sentences):
                entity_start = entity.get("start", -1)
                if sent["start"] <= entity_start < sent["end"]: process_end_events_map[i] = entity; break
    for pair in agent_task_pairs:
        sent_idx = pair.get("sentence_idx")
        if sent_idx is not None and sent_idx in process_end_events_map:
            if "process_end_event" not in pair: pair["process_end_event"] = process_end_events_map[sent_idx]
    return agent_task_pairs

def has_parallel_keywords(text: str) -> bool:
    if not nlp_md: return False
    matcher = Matcher(nlp_md.vocab)
    patterns = [
        [{"LOWER": {"IN": ["in", "at"]}}, {"LOWER": "the"}, {"LOWER": {"IN": ["meantime", "same"]}}, {"LOWER": "time"}],
        [{"LOWER": "meanwhile"}], [{"LOWER": "while"}], [{"LOWER": "in"}, {"LOWER": "parallel"}],
        [{"LOWER": "parallel"}, {"LOWER": {"IN": ["paths", "activities", "tasks", "execution"]}}],
        [{"LOWER": "concurrently"}], [{"LOWER": "simultaneously"}] ]
    matcher.add("PARALLEL", patterns); doc = nlp_md(text.lower()); return len(matcher(doc)) > 0

def find_sentences_with_loop_keywords(sentences: list[dict]) -> list[dict]:
    if not nlp_md: return []
    matcher = Matcher(nlp_md.vocab)
    patterns = [ [{"LOWER": "again"}], [{"LOWER": {"IN": ["repeat", "repeats"]}}], [{"LOWER": {"IN": ["iterate", "iterates"]}}],
        [{"LOWER": "loop"}, {"LOWER": "back"}], [{"LOWER": "until"}, {"POS": "NOUN"}], [{"LOWER": "go", "LOWER": "back", "LOWER": "to"}] ]
    matcher.add("LOOP", patterns); return [sent for sent in sentences if len(matcher(nlp_md(sent["sentence"].lower()))) > 0]

def add_task_ids(agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]) -> list[dict]:
    task_id_counter = 0
    loop_sentence_indices = {sent['start'] for sent in loop_sentences}
    for pair in agent_task_pairs:
        if "task" in pair and isinstance(pair["task"], dict):
            task = pair["task"]; task_in_loop = False
            for sent in sentences:
                task_start = task.get("start", -1)
                if task_start != -1 and sent["start"] <= task_start < sent["end"]:
                    if sent['start'] in loop_sentence_indices: task_in_loop = True; break
            if not task_in_loop: task["task_id"] = f"T{task_id_counter}"; task_id_counter += 1
    return agent_task_pairs

def add_loops(agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]) -> list[dict]:
    tasks_with_ids = { pair["task"]["task_id"]: pair["task"] for pair in agent_task_pairs if "task" in pair and isinstance(pair["task"], dict) and "task_id" in pair["task"] }
    loop_sentence_indices = {sent['start'] for sent in loop_sentences}
    processed_elements = []
    for pair in agent_task_pairs:
        if "task" in pair and isinstance(pair["task"], dict):
            task = pair["task"]; task_sentence_start = -1
            task_start = task.get("start", -1)
            if task_start != -1:
                for sent in sentences:
                    if sent["start"] <= task_start < sent["end"]: task_sentence_start = sent['start']; break
            if task_sentence_start in loop_sentence_indices:
                 previous_tasks_list = [t for t in list(tasks_with_ids.values()) if t.get("start") != task_start]
                 previous_task_to_loop_to = find_previous_task(previous_tasks_list, task)
                 if previous_task_to_loop_to and "task_id" in previous_task_to_loop_to:
                     loop_element = { "type": "loop", "go_to": previous_task_to_loop_to["task_id"], "start": task.get("start"), "end": task.get("end"),
                         "sentence_idx": pair.get("sentence_idx"), "original_loop_task": task, "original_loop_agent": pair.get("agent") }
                     processed_elements.append(loop_element)
                     print(f"  Created loop element: looping from '{task.get('word')}' back to '{previous_task_to_loop_to.get('word')}' (ID: {previous_task_to_loop_to['task_id']})")
                 else:
                     print(f"{Fore.YELLOW}Warning: Could not find suitable previous task with ID to loop back to for task '{task.get('word', '')}'. Skipping loop creation, keeping task.{Fore.RESET}")
                     processed_elements.append(pair)
            else: processed_elements.append(pair)
        else: processed_elements.append(pair)
    return processed_elements

def find_previous_task(previous_tasks: list[dict], task: dict) -> dict | None:
    previous_tasks_with_ids = [t for t in previous_tasks if "task_id" in t]
    if not previous_tasks_with_ids: return None
    previous_tasks_str = "\n".join([f"{t['task_id']}: {t.get('word', '')}" for t in previous_tasks_with_ids])
    task_word = task.get('word', '')
    if not task_word: return previous_tasks_with_ids[-1] if previous_tasks_with_ids else None
    try:
        previous_task_text_suggestion = prompts.find_previous_task(task_word, previous_tasks_str)
        if not previous_task_text_suggestion:
            print(f"{Fore.YELLOW}Warning: LLM did not suggest a previous task for '{task_word}'. Falling back to last task.{Fore.RESET}")
            return previous_tasks_with_ids[-1]
        choices = {t['task_id']: t.get('word', '') for t in previous_tasks_with_ids}
        best_match_result = process.extractOne(previous_task_text_suggestion, choices, scorer=fuzz.token_sort_ratio)
        highest_similarity_task = None; highest_similarity = -1
        if best_match_result:
            matched_word, score, task_id = best_match_result; highest_similarity = score
            highest_similarity_task = next((t for t in previous_tasks_with_ids if t['task_id'] == task_id), None)
        similarity_threshold = 65
        if highest_similarity < similarity_threshold:
             print(f"{Fore.YELLOW}Warning: Low similarity ({highest_similarity}%) between DeepSeek suggestion '{previous_task_text_suggestion}' and potential loop targets for '{task_word}'. Loop might be incorrect. Falling back to last task.{Fore.RESET}")
             return previous_tasks_with_ids[-1]
        return highest_similarity_task
    except Exception as e:
        print(f"{Fore.RED}Error during DeepSeek call or fuzzy matching in find_previous_task: {e}{Fore.RESET}")
        traceback.print_exc()
        return previous_tasks_with_ids[-1] if previous_tasks_with_ids else None

# --- ИЗМЕНЕННАЯ ФУНКЦИЯ extract_exclusive_gateways С ФИЛЬТРАЦИЕЙ ВЛОЖЕННОСТИ ---
def extract_exclusive_gateways(process_description: str, conditions: list) -> list:
    """
    Извлекает эксклюзивные шлюзы, фильтрует вложенные, ассоциирует условия.
    """
    global sents_data
    if not conditions: return []
    conditions.sort(key=lambda x: x.get('start', float('inf')))
    first_condition_start = conditions[0].get("start")
    if first_condition_start is None: return []

    exclusive_gateway_text_full = process_description[first_condition_start:]
    response = ""
    try:
        if len(conditions) == 2: response = prompts.extract_exclusive_gateways_2_conditions(exclusive_gateway_text_full)
        elif len(conditions) > 2: response = prompts.extract_exclusive_gateways(exclusive_gateway_text_full)
        else: return []
    except Exception as e: print(f"{Fore.RED}Error during DeepSeek call (extract_exclusive_gateways): {e}{Fore.RESET}"); return []

    pattern = r"Exclusive gateway \d+:\s*(.*?)(?=(?:Exclusive gateway \d+:|$))"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    gateway_texts_from_llm = [m.strip() for m in matches if m.strip()]
    if not gateway_texts_from_llm: print(f"{Fore.YELLOW}Warning: Could not parse gateway spans from LLM.{Fore.RESET}"); return []

    # --- Находим индексы и создаем кандидатов ---
    gateway_candidates = []
    try:
        gateway_indices_list = get_indices(gateway_texts_from_llm, process_description)
        for idx, text in enumerate(gateway_texts_from_llm):
            indices = gateway_indices_list[idx]
            if indices: gateway_candidates.append({"text": text, "start": indices["start"], "end": indices["end"], "original_llm_idx": idx})
            else: print(f"{Fore.YELLOW}Warning: Could not find indices for LLM gateway text {idx}.{Fore.RESET}")
    except Exception as e: print(f"{Fore.RED}Error finding indices for LLM gateway texts: {e}{Fore.RESET}"); return []

    if not gateway_candidates: return []

    # --- Фильтрация Вложенных Шлюзов ---
    gateway_candidates.sort(key=lambda x: (x["start"], -x["end"]))
    filtered_candidates = []
    indices_to_remove = set()
    for i in range(len(gateway_candidates)):
        if i in indices_to_remove: continue
        for j in range(len(gateway_candidates)):
            if i == j or j in indices_to_remove: continue
            # Если j вложен в i
            if gateway_candidates[i]["start"] <= gateway_candidates[j]["start"] and gateway_candidates[i]["end"] >= gateway_candidates[j]["end"]:
                indices_to_remove.add(j)
            # Если i вложен в j
            elif gateway_candidates[j]["start"] <= gateway_candidates[i]["start"] and gateway_candidates[j]["end"] >= gateway_candidates[i]["end"]:
                indices_to_remove.add(i)
                break # Если i вложен, дальше для него проверять не нужно
        if i not in indices_to_remove:
             filtered_candidates.append(gateway_candidates[i])

    print(f"  Filtered {len(gateway_candidates) - len(filtered_candidates)} potentially nested/overlapping gateway candidates.")
    if not filtered_candidates: return []


    # --- Ассоциация условий с отфильтрованными кандидатами ---
    exclusive_gateways = []
    gateway_id_counter = 0
    original_condition_entities = {c.get('start'): c for c in conditions if c.get('start') is not None}
    assigned_condition_starts = set()

    # Сортируем отфильтрованных кандидатов по порядку появления в тексте
    filtered_candidates.sort(key=lambda x: x["start"])

    for candidate in filtered_candidates:
        gw_start, gw_end = candidate['start'], candidate['end']
        associated_conditions_for_gw = []
        # Ищем условия СТРОГО внутри диапазона и еще не присвоенные
        for cond_start, condition_entity in original_condition_entities.items():
            if cond_start in assigned_condition_starts: continue
            cond_end = condition_entity.get("end")
            if cond_end is not None and gw_start <= cond_start < cond_end <= gw_end:
                 associated_conditions_for_gw.append(condition_entity)

        if associated_conditions_for_gw:
            gw_id = f"EG{gateway_id_counter}"; gateway_id_counter += 1
            condition_words = [cond.get('word') for cond in associated_conditions_for_gw if cond.get('word')]
            condition_indices = [{"start": cond.get("start"), "end": cond.get("end")} for cond in associated_conditions_for_gw]
            suggested_name = prompts.suggest_gateway_name(candidate['text'])
            print(f"Suggested name for Gateway {gw_id}: '{suggested_name}'")
            exclusive_gateways.append({
                "id": gw_id, "name": suggested_name, "conditions": condition_words,
                "start": candidate["start"], "end": candidate["end"], "paths": condition_indices
            })
            for cond_entity in associated_conditions_for_gw: assigned_condition_starts.add(cond_entity.get('start'))
        # else: # Не печатаем это сообщение, т.к. фильтрация могла убрать шлюз, к которому относилось условие
             # print(f"{Fore.CYAN}Info: Skipping filtered gateway candidate ('{candidate['text'][:50]}...') as no unassigned conditions were found within its span.{Fore.RESET}")

    exclusive_gateways.sort(key=lambda x: x.get('start', float('inf')))
    print(f"Processed {len(exclusive_gateways)} valid exclusive gateways after filtering and strict association.")
    return exclusive_gateways

def handle_text_with_conditions(agent_task_pairs: list, conditions: list, sents_data: list, process_desc: str) -> tuple[list, list]:
    """
    Handles processing steps related to conditions: extracting exclusive gateways.
    """
    exclusive_gateway_data = extract_exclusive_gateways(process_desc, conditions)
    return agent_task_pairs, exclusive_gateway_data

# --- Функции should_resolve_coreferences, extract_all_entities, get_indices (без изменений) ---
def should_resolve_coreferences(text: str) -> bool:
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "theirs"}
    words = re.findall(r'\b\w+\b', text.lower())
    return any(word in pronouns for word in words)

def extract_all_entities(data: list | None, min_score: float) -> tuple:
    if data is None: return ([], [], [], [])
    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data, min_score)
    tasks = extract_entities("TASK", data, min_score)
    conditions = extract_entities("CONDITION", data, min_score)
    process_info = extract_entities("PROCESS_INFO", data, min_score)
    return (agents, tasks, conditions, process_info)

def get_indices(strings_to_find: list[str], text: str) -> list[dict | None]:
    """
    Finds start and end indices for a list of strings within a larger text.
    """
    results = []
    text_lower = text.lower()
    for string_to_find in strings_to_find:
        if not string_to_find or not isinstance(string_to_find, str):
            print(f"{Fore.YELLOW}Warning: Attempted to find indices for invalid string: {string_to_find}. Skipping.{Fore.RESET}")
            results.append(None); continue
        string_lower = string_to_find.lower(); best_match_info = None
        try: # Exact Match
            start_index = text_lower.index(string_lower); end_index = start_index + len(string_lower);
            best_match_info = {"start": start_index, "end": end_index, "score": 100}
        except ValueError: # Fuzzy Match
            try:
                max_len_diff = 15; search_range = len(text) - len(string_to_find) + max_len_diff; step = max(1, search_range // 1000);
                choices = [text[i : i + len(string_to_find) + max_len_diff] for i in range(0, search_range, step)]
                if choices:
                    match_result = process.extractOne(string_to_find, choices, scorer=fuzz.partial_ratio)
                    if match_result:
                        match, score = match_result; fuzzy_threshold = 85
                        if score >= fuzzy_threshold:
                            fuzzy_start = text.find(match)
                            if fuzzy_start != -1:
                                best_match_info = {"start": fuzzy_start, "end": fuzzy_start + len(match), "score": score}
                                print(f"{Fore.CYAN}Info: Fuzzy matched '{string_to_find}' to '{match}' (Score: {score}) at index {fuzzy_start}.{Fore.RESET}")
                            else: print(f"{Fore.YELLOW}Warning: Could not locate fuzzy match '{match}' in text.{Fore.RESET}")
            except Exception as e: print(f"{Fore.RED}Error during fuzzy search for '{string_to_find}': {e}{Fore.RESET}")
        if best_match_info: results.append({"start": best_match_info["start"], "end": best_match_info["end"]})
        else: print(f"{Fore.YELLOW}Warning: Could not find indices for '{string_to_find}'.{Fore.RESET}"); results.append(None)
    return results

# --- Функции get_parallel_paths, get_parallel_gateways, handle_text_with_parallel_keywords (без изменений) ---
def get_parallel_paths(parallel_gateway_text: str, process_description: str) -> list[dict] | None:
    try:
        num_str = prompts.number_of_parallel_paths(parallel_gateway_text); num = int(num_str)
    except ValueError: print(f"{Fore.YELLOW}Warning: Could not determine number of parallel paths from '{num_str}'. Assuming 2.{Fore.RESET}"); num = 2
    except Exception as e: print(f"{Fore.RED}Error during DeepSeek call for number of paths: {e}{Fore.RESET}"); return None
    if num > 3: num = 3;
    if num <= 1: return None
    paths_text = ""; response_paths = []
    try:
        if num == 2: paths_text = prompts.extract_2_parallel_paths(parallel_gateway_text)
        elif num == 3: paths_text = prompts.extract_3_parallel_paths(parallel_gateway_text)
        response_paths = [s.strip() for s in paths_text.split("&&") if s.strip()]
    except Exception as e: print(f"{Fore.RED}Error during DeepSeek call for extracting paths: {e}{Fore.RESET}"); return None
    if len(response_paths) != num and response_paths: print(f"{Fore.YELLOW}Warning: Expected {num} paths, parsed {len(response_paths)}. Using parsed.{Fore.RESET}")
    elif not response_paths: print(f"{Fore.YELLOW}Warning: LLM did not return parseable parallel paths.{Fore.RESET}"); return None
    try:
         indices = get_indices(response_paths, process_description)
         if any(idx is None for idx in indices):
             failed_paths = [p for p, idx in zip(response_paths, indices) if idx is None]
             print(f"{Fore.RED}Error: Could not find indices for all parallel paths extracted by LLM: {failed_paths}. Skipping path data.{Fore.RESET}")
             return None
         return indices
    except Exception as e: print(f"{Fore.RED}Error finding indices for parallel paths: {e}{Fore.RESET}"); return None

def get_parallel_gateways(text: str) -> list[dict]:
    """ Extracts text spans corresponding to parallel gateways using LLM. """
    try: response = prompts.extract_parallel_gateways(text)
    except Exception as e: print(f"{Fore.RED}Error during DeepSeek call in get_parallel_gateways: {e}{Fore.RESET}"); return []
    pattern = r"Parallel gateway \d+:\s*(.*?)(?=(?:Parallel gateway \d+:|$))"; matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    gateway_texts = [s.strip() for s in matches if s.strip()]
    if not gateway_texts: return []
    try:
        indices = get_indices(gateway_texts, text); valid_gateway_regions = []
        for text_span, index_data in zip(gateway_texts, indices):
             if index_data: valid_gateway_regions.append(index_data)
             else: print(f"{Fore.YELLOW}Warning: Could not find indices for parallel gateway text: '{text_span[:50]}...'{Fore.RESET}")
        return valid_gateway_regions
    except Exception as e: print(f"{Fore.RED}Error finding indices for parallel gateway texts: {e}{Fore.RESET}"); return []

def handle_text_with_parallel_keywords( process_description: str, agent_task_pairs: list[dict], sents_data: list[dict] ) -> list[dict]:
    """ Identifies parallel gateway regions, extracts paths, suggests names, etc. """
    parallel_gateways_data = []; parallel_gateway_id_counter = 0
    gateway_region_indices = get_parallel_gateways(process_description)
    if not gateway_region_indices: return []
    elements_to_remove_indices = set(); atp_indices_map = {id(pair): i for i, pair in enumerate(agent_task_pairs)}
    for region_idx, region_indices in enumerate(gateway_region_indices):
        gateway_id = f"PG{parallel_gateway_id_counter}"; parallel_gateway_id_counter += 1
        if not region_indices or 'start' not in region_indices or 'end' not in region_indices: print(f"{Fore.YELLOW}Warning: Invalid region indices for PG {gateway_id}. Skipping.{Fore.RESET}"); continue
        gateway_text = process_description[region_indices["start"] : region_indices["end"]]
        num_sentences_spanned = count_sentences_spanned(sents_data, region_indices)
        num_atp_in_region = num_of_agent_task_pairs_in_range(agent_task_pairs, region_indices)
        pg_path_indices = []; suggested_name = "Parallel Gateway"
        if num_sentences_spanned <= 1 and num_atp_in_region <= 1: # Case 1: Single Sentence Expansion
            print(f"PG {gateway_id}: Handling potential parallel tasks in single sentence.")
            sentence_text = get_sentence_text(sents_data, region_indices)
            if not sentence_text: print(f"{Fore.YELLOW}Warning: Could not get sentence text for region {gateway_id}. Skipping.{Fore.RESET}"); continue
            parallel_tasks_text = [];
            try: response = prompts.extract_parallel_tasks(sentence_text); parallel_tasks_text = extract_tasks(response)
            except Exception as e: print(f"{Fore.RED}Error extracting parallel tasks for {gateway_id}: {e}{Fore.RESET}"); continue
            if len(parallel_tasks_text) <= 1: print(f"{Fore.YELLOW}Warning: Did not extract multiple tasks for {gateway_id}. Skipping.{Fore.RESET}"); continue
            suggested_name = prompts.suggest_gateway_name(gateway_text or sentence_text); print(f"Suggested name for PG {gateway_id}: '{suggested_name}'")
            original_atp_identity = get_agent_task_pair_identity(agent_task_pairs, region_indices); original_atp = None; original_atp_index = -1
            if original_atp_identity: original_atp = next((p for p in agent_task_pairs if id(p) == original_atp_identity), None);
            if original_atp: original_atp_index = atp_indices_map.get(original_atp_identity, -1)
            if original_atp_index != -1: elements_to_remove_indices.add(original_atp_index)
            new_pairs_for_region = []; task_start_offset = region_indices.get('start', 0)
            for i, task_word in enumerate(parallel_tasks_text):
                 current_task_start = task_start_offset + i * 2; current_task_end = current_task_start + 1
                 new_task_entity = {"entity_group": "TASK", "start": current_task_start, "end": current_task_end, "word": task_word, "score": 0.99, "task_id": f"{gateway_id}_T{i}" }
                 sent_idx = original_atp.get('sentence_idx', -1) if original_atp else -1
                 if sent_idx == -1 and sents_data: sent_idx = next((i for i, s in enumerate(sents_data) if s['start'] <= task_start_offset < s['end']), -1)
                 agent_info = original_atp.get('agent') if original_atp else {"original_word": "Unknown", "resolved_word": "Unknown", "entity": None}
                 new_atp = {"agent": agent_info, "task": new_task_entity, "sentence_idx": sent_idx}; new_pairs_for_region.append(new_atp)
                 pg_path_indices.append({"start": current_task_start, "end": current_task_end, "task_id_ref": new_task_entity["task_id"]})
            insert_pos = original_atp_index if original_atp_index != -1 else len(agent_task_pairs)
            for item in reversed(new_pairs_for_region): agent_task_pairs.insert(insert_pos, item)
            atp_indices_map = {id(pair): i for i, pair in enumerate(agent_task_pairs)}
            if pg_path_indices:
                 gateway_start = pg_path_indices[0]["start"]; gateway_end = pg_path_indices[-1]["end"]
                 gateway_data = {"id": gateway_id, "name": suggested_name, "start": gateway_start, "end": gateway_end, "paths": pg_path_indices, "type": "single_sentence_expansion"}; parallel_gateways_data.append(gateway_data)
        else: # Case 2: Multi-Sentence Region
            path_indices = get_parallel_paths(gateway_text, process_description)
            if path_indices:
                suggested_name = prompts.suggest_gateway_name(gateway_text); print(f"Suggested name for PG {gateway_id}: '{suggested_name}'")
                gateway_data = { "id": gateway_id, "name": suggested_name, "start": region_indices["start"], "end": region_indices["end"], "paths": path_indices, "type": "multi_sentence_region" }; parallel_gateways_data.append(gateway_data)
            else: print(f"{Fore.YELLOW}Warning: Could not determine paths for PG {gateway_id}. Skipping.{Fore.RESET}")
    if elements_to_remove_indices:
        sorted_indices_to_remove = sorted(list(elements_to_remove_indices), reverse=True)
        print(f"Attempting to remove original ATPs at indices: {sorted_indices_to_remove}")
        for index_to_remove in sorted_indices_to_remove:
            if 0 <= index_to_remove < len(agent_task_pairs):
                try: removed_pair = agent_task_pairs.pop(index_to_remove)
                except IndexError: print(f"{Fore.RED}Error: Could not remove ATP at index {index_to_remove}.{Fore.RESET}")
            else: print(f"{Fore.YELLOW}Warning: Index {index_to_remove} out of bounds.{Fore.RESET}")
        atp_indices_map = {id(pair): i for i, pair in enumerate(agent_task_pairs)}
    gateways_to_add = [] # Nested Gateways
    parallel_gateways_data.sort(key=lambda x: x.get('start', float('inf')))
    for gateway in parallel_gateways_data:
        if "parallel_parent" in gateway or gateway.get("type") == "single_sentence_expansion": continue
        if "paths" in gateway and gateway["paths"]:
            for path_idx, path in enumerate(gateway["paths"]):
                 path_start = path.get("start"); path_end = path.get("end")
                 if path_start is None or path_end is None: continue
                 path_text = process_description[path_start : path_end]
                 if has_parallel_keywords(path_text):
                     nested_path_indices = get_parallel_paths(path_text, process_description)
                     if nested_path_indices:
                         nested_gateway_id = f"PG{parallel_gateway_id_counter}"; parallel_gateway_id_counter += 1
                         nested_suggested_name = prompts.suggest_gateway_name(path_text); print(f"Suggested name for Nested PG {nested_gateway_id}: '{nested_suggested_name}'")
                         if nested_path_indices and all(isinstance(p, dict) for p in nested_path_indices):
                            nested_starts = [p.get("start") for p in nested_path_indices if p.get("start") is not None]; nested_ends = [p.get("end") for p in nested_path_indices if p.get("end") is not None]
                            if nested_starts and nested_ends:
                                nested_start = min(nested_starts); nested_end = max(nested_ends)
                                nested_gateway_data = { "id": nested_gateway_id, "name": nested_suggested_name, "start": nested_start, "end": nested_end, "paths": nested_path_indices, "parallel_parent": gateway["id"], "parallel_parent_path_idx": path_idx, "type": "nested_region" }; gateways_to_add.append(nested_gateway_data)
                            else: print(f"{Fore.YELLOW}Warning: Could not get start/end for nested PG {nested_gateway_id}. Skipping.{Fore.RESET}")
                         else: print(f"{Fore.YELLOW}Warning: Invalid nested path data for PG {nested_gateway_id}. Skipping.{Fore.RESET}")
    parallel_gateways_data.extend(gateways_to_add); parallel_gateways_data.sort(key=lambda x: x.get('start', float('inf')))
    print(f"Processed {len(parallel_gateways_data)} parallel gateways (including nested).")
    return parallel_gateways_data

def num_of_agent_task_pairs_in_range(agent_task_pairs: list[dict], indices: dict[str, int]) -> int:
    """Counts how many agent_task_pairs' tasks overlap with the given indices range."""
    count = 0; start_idx = indices.get('start', -1); end_idx = indices.get('end', -1)
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx: return 0
    for pair in agent_task_pairs:
        if "task" in pair and isinstance(pair["task"], dict):
            task_start = pair["task"].get("start", -1); task_end = pair["task"].get("end", -1)
            if task_start != -1 and task_end != -1 and task_start < task_end:
                 if max(task_start, start_idx) < min(task_end, end_idx): count += 1
    return count

def get_agent_task_pair_identity(agent_task_pairs: list[dict], indices: dict[str, int]) -> int | None:
    """Возвращает id() объекта agent_task_pair, задача которого попадает в диапазон."""
    start_idx = indices.get('start', -1); end_idx = indices.get('end', -1)
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx: return None
    for pair in agent_task_pairs:
        if "task" in pair and isinstance(pair["task"], dict):
            task_start = pair["task"].get("start", -1); task_end = pair["task"].get("end", -1)
            if task_start != -1 and task_end != -1 and task_start < task_end:
                if max(task_start, start_idx) < min(task_end, end_idx): return id(pair)
    return None

def count_sentences_spanned(sentence_data: list[dict], indices: dict[str, int], buffer: int = 0) -> int:
    """Counts how many sentences overlap with the given indices range."""
    count = 0; idx_start = indices.get('start', -1); idx_end = indices.get('end', -1)
    if buffer > 0: idx_start += buffer; idx_end -= buffer
    if idx_start == -1 or idx_end == -1 or idx_start >= idx_end: return 0
    for sentence_info in sentence_data:
        sent_start = sentence_info.get('start', -1); sent_end = sentence_info.get('end', -1)
        if sent_start != -1 and sent_end != -1 and sent_start < sent_end:
            if max(sent_start, idx_start) < min(sent_end, idx_end): count += 1
    return count

def get_sentence_text(sentence_data: list[dict], indices: dict[str, int]) -> str | None:
    """Finds the text of the first sentence overlapping with the given indices."""
    idx_start = indices.get('start', -1); idx_end = indices.get('end', -1)
    if idx_start == -1 or idx_end == -1 or idx_start >= idx_end: return None
    for sentence_info in sentence_data:
        sent_start = sentence_info.get('start', -1); sent_end = sentence_info.get('end', -1)
        if sent_start != -1 and sent_end != -1 and sent_start < sent_end:
             if max(sent_start, idx_start) < min(sent_end, idx_end): return sentence_info.get('sentence')
    return None

def extract_tasks(model_response: str) -> list[str]:
    """Extracts task descriptions from LLM response formatted as 'Task N: ...'"""
    if not model_response: return []
    pattern = r"Task \d+:\s*(.*?)(?=(?:Task \d+:|$))"; matches = re.findall(pattern, model_response, re.DOTALL | re.IGNORECASE)
    return [s.strip() for s in matches if s.strip()]


# --- Основная функция process_text ---
def process_text(text: str) -> dict | None:
    log_dir = "./output_logs"; clear_folder(log_dir)
    print(f"\n{Fore.CYAN}--- Starting BPMN Processing ---{Fore.RESET}")
    print(f"\nInput text:\n{text}\n"); original_text = text

    clusters = None
    if coref_model and should_resolve_coreferences(original_text):
        print("Attempting coreference resolution...\n"); coref_info = get_coref_info(original_text, print_clusters=False)
        if coref_info: clusters = coref_info.get('clusters_str'); print(f"{Fore.GREEN}Coreference resolution successful. Found {len(clusters) if clusters else 0} clusters.{Fore.RESET}\n")
        else: print(f"{Fore.YELLOW}Coreference resolution failed or returned no info.{Fore.RESET}\n")
    else:
        if not coref_model: print(f"{Fore.YELLOW}Coref model not loaded. Skipping coreference resolution.{Fore.RESET}\n")
        else: print("No relevant pronouns found, skipping coreference resolution.\n")

    bpmn_entities = extract_bpmn_data(original_text)
    if bpmn_entities is None: print(f"{Fore.RED}Failed to extract BPMN entities. Aborting process.{Fore.RESET}"); return None
    bpmn_entities = fix_bpmn_data(bpmn_entities)
    agents, tasks, conditions, process_info = extract_all_entities(bpmn_entities, min_score=0.75)
    print(f"Extracted: {len(agents)} Agents, {len(tasks)} Tasks, {len(conditions)} Conditions, {len(process_info)} Process Infos\n")
    parallel_gateway_data = []; exclusive_gateway_data = []

    sentence_data = create_sentence_data(original_text)
    if not sentence_data: print(f"{Fore.YELLOW}Warning: Could not extract sentence data.{Fore.RESET}")

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sentence_data, clusters)

    if has_parallel_keywords(original_text):
        print("Parallel keywords detected. Handling parallel gateways...\n")
        parallel_gateway_data = handle_text_with_parallel_keywords(original_text, agent_task_pairs, sentence_data)
    else: print("No parallel keywords detected.\n")

    if conditions:
        print("Conditions detected. Handling exclusive gateways...\n")
        # ИСПРАВЛЕНИЕ: Передаем оригинальный agent_task_pairs, так как handle_text_with_conditions теперь его не меняет
        _, exclusive_gateway_data = handle_text_with_conditions(agent_task_pairs, conditions, sentence_data, original_text)
    else: print("No conditions detected.\n")

    classified_process_info = []
    if process_info:
        print("Handling PROCESS_INFO entities...\n")
        classified_process_info = batch_classify_process_info(process_info)
        agent_task_pairs = add_process_end_events(agent_task_pairs, sentence_data, classified_process_info)
    else: print("No PROCESS_INFO entities found.\n")

    print("Handling loops...\n")
    loop_sentences = find_sentences_with_loop_keywords(sentence_data)
    if loop_sentences: print(f"Found {len(loop_sentences)} potential loop sentences.")
    agent_task_pairs = add_task_ids(agent_task_pairs, sentence_data, loop_sentences)
    agent_task_pairs_or_loops = add_loops(agent_task_pairs, sentence_data, loop_sentences)

    print("\nExtracting unique agents...")
    unique_agents = []; seen_agents = set()
    for p_or_l in agent_task_pairs_or_loops:
        agent_info = None
        if isinstance(p_or_l, dict):
             if "agent" in p_or_l: agent_info = p_or_l.get('agent')
             elif "original_loop_agent" in p_or_l: agent_info = p_or_l.get('original_loop_agent')
        if isinstance(agent_info, dict) and 'resolved_word' in agent_info:
            agent_name = agent_info['resolved_word']
            if agent_name and isinstance(agent_name, str) and agent_name.lower() not in seen_agents:
                 unique_agents.append(agent_name); seen_agents.add(agent_name.lower())
    if not unique_agents:
        print(f"{Fore.YELLOW}Warning: No resolved agents found. Trying fallback from initial NER.{Fore.RESET}")
        unique_agents = []; seen_agents = set()
        for agent_entity in agents:
            agent_name = agent_entity.get('word')
            if agent_name and isinstance(agent_name, str) and agent_name.lower() not in seen_agents:
                unique_agents.append(agent_name); seen_agents.add(agent_name.lower())
        if not unique_agents: print(f"{Fore.YELLOW}Warning: Fallback also found no agents.{Fore.RESET}")
    print("Unique Agents Found:", unique_agents)

    print("\nCreating final BPMN JSON structure...\n")
    final_bpmn_json = create_bpmn_structure(
        agent_task_pairs_or_loops=agent_task_pairs_or_loops,
        parallel_gateways=parallel_gateway_data,
        exclusive_gateways=exclusive_gateway_data,
        process_info_entities=classified_process_info,
        unique_agents=unique_agents,
        original_text=original_text )

    if final_bpmn_json is None: print(f"{Fore.RED}Failed to create BPMN JSON structure.{Fore.RESET}"); return None
    print("BPMN JSON Structure created successfully.")

    output_dir = "./output_bpmn"
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir)
        except OSError as e: print(f"{Fore.RED}Error creating output directory {output_dir}: {e}{Fore.RESET}"); output_dir = "."
    output_filename = os.path.join(output_dir, "bpmn_final_structure.json")
    try: write_to_file(output_filename, final_bpmn_json); print(f"\nFinal BPMN JSON structure saved to {output_filename}")
    except Exception as e: print(f"{Fore.RED}Error saving final BPMN JSON to {output_filename}: {e}{Fore.RESET}")

    print(f"\n{Fore.CYAN}--- BPMN Processing Finished ---{Fore.RESET}")
    return final_bpmn_json

if __name__ == "__main__":
    example_text = (
        "The company receives the order from the customer."
        " If the product is out of stock, the customer receives a notification that the order cannot be fulfilled."
        " If the product is in stock and the payment succeeds, the company processes and ships the order."
        " If the product is in stock, but the payment fails, the customer receives a notification that the order cannot be processed."
    )
    print(f"\n--- Running processing for example text ---"); print(f"Text: {example_text}"); print("-------------------------------------------\n")
    result_json = process_text(example_text)
    if result_json:
        print("\n--- Processing Complete ---")
        try:
            definitions = result_json.get('definitions', {}); process_data = definitions.get('process', {}); flow_elements = process_data.get('flowElements', []); sequence_flows = process_data.get('sequenceFlows', []); diagrams_list = result_json.get('diagrams', [{}]); plane = diagrams_list[0].get('plane', {}) if diagrams_list else {}; shapes_edges = plane.get('planeElement', [])
            print(f"Generated {len(flow_elements)} flow elements and {len(sequence_flows)} sequence flows.")
            shape_count = sum(1 for el in shapes_edges if 'bounds' in el); edge_count = sum(1 for el in shapes_edges if 'waypoints' in el); print(f"Generated {shape_count} shapes and {edge_count} edges visually.")
        except Exception as e: print(f"Could not retrieve element counts from result: {e}"); traceback.print_exc()
    else: print("\n--- Processing Failed ---")

# --- END OF FILE process_bpmn_data.py ---