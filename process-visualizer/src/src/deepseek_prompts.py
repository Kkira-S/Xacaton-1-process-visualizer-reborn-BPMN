# --- START OF FILE deepseek_prompts.py ---
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import traceback
from colorama import Fore, init # Для цветного вывода предупреждений

init(autoreset=True)

# --- Инициализация Модели ---
model_loaded = False
tokenizer = None
model = None
DEVICE = "cpu"
try:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    print(f"DeepSeek model loaded successfully on {DEVICE}.")
    model_loaded = True
except Exception as e:
    print(f"{Fore.RED}Error loading DeepSeek model: {e}{Fore.RESET}")
    print(f"{Fore.YELLOW}LLM prompts will return default values or empty strings.{Fore.RESET}")

# --- Системные сообщения ---
SYSTEM_MSG_BPMN_EXPERT = ("You are a highly experienced business process modelling expert, specializing in BPMN modelling. "
                          "You will be provided with descriptions of business processes or parts of them. "
                          "Your answers must be accurate, concise, and strictly follow the requested format.")

# --- Вспомогательная функция для генерации ---
def _generate_llm_response(messages: list, max_new_tokens: int, temperature: float = 0.0) -> str:
    """Вспомогательная функция для генерации ответа LLM с обработкой ошибок."""
    if not model_loaded:
        print(f"{Fore.YELLOW}Warning: LLM model not loaded. Returning empty string.{Fore.RESET}")
        return "" # Заглушка

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Ограничиваем длину входного промпта
        inputs = tokenizer(prompt, return_tensors="pt", max_length=3072, truncation=True).to(DEVICE) # Увеличим немного max_length

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if not decoded_outputs: return ""

        # Более надежное извлечение ответа модели
        # Ищем последний 'assistant\n' или подобный маркер конца промпта
        prompt_end_marker = "assistant\n" # Стандартный маркер для chat template
        response_start_index = decoded_outputs[0].rfind(prompt_end_marker)
        if response_start_index != -1:
             response_part = decoded_outputs[0][response_start_index + len(prompt_end_marker):].strip()
        else:
             # Если маркер не найден (менее вероятно), пытаемся отрезать исходный user prompt
             last_user_message = messages[-1]['content']
             response_part = decoded_outputs[0].split(last_user_message)[-1].strip()

        return response_part

    except Exception as e:
        print(f"{Fore.RED}Error during LLM generation: {e}{Fore.RESET}")
        traceback.print_exc()
        return ""

# --- Основные функции промптов ---

def suggest_gateway_name(gateway_text: str) -> str:
    """
    Предлагает короткое, вопросительное (если применимо) имя для шлюза.
    """
    if not model_loaded: return "Gateway" # Заглушка

    cleaned_text = ' '.join(gateway_text.split())
    max_input_length = 512
    if len(cleaned_text) > max_input_length: cleaned_text = cleaned_text[:max_input_length] + "..."

    system_prompt = (
        "You are a BPMN expert. Given text describing conditions, decisions, or parallel actions at a gateway, "
        "suggest a concise, meaningful name (2-5 words). If it's a decision, end with '?'. "
        "If it's parallel execution, use a verb phrase. Output ONLY the suggested name."
    )
    user_prompt = (
        "Text: 'If the customer chooses to finance... If the customer chooses to pay in cash...'\n"
        "Name: Payment Method?\n\n"
        "Text: 'If the item is in stock... If the item is out of stock...'\n"
        "Name: Item in Stock?\n\n"
        "Text: 'If the design is approved... If not...'\n"
        "Name: Design Approved?\n\n"
        "Text: 'one team verifies employment while another verifies income simultaneously.'\n"
        "Name: Verify Applicant Info\n\n" # Изменен пример
        "Text: 'Send mail while preparing documents.'\n"
        "Name: Send Mail & Prepare Docs\n\n" # Добавлен пример
        f"Text: '{cleaned_text}'\n"
        "Name:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = _generate_llm_response(messages, max_new_tokens=15, temperature=0.1)
    response = response.split('\n')[0].strip().replace('"', '')

    # Пост-обработка для добавления '?' (можно улучшить логику)
    decision_keywords = ["if", "whether", "choose", "decide", "option", "case"]
    if any(keyword in gateway_text.lower() for keyword in decision_keywords) and "?" not in response:
        if not response.endswith('?'): response += '?'

    return response if response else "Gateway"


def extract_exclusive_gateways(process_description: str) -> str:
    """
    Извлекает текст, относящийся к эксклюзивным шлюзам.
    """
    system_msg = (
        "You are a BPMN expert. Extract the text spans belonging to **exclusive** decision points (gateways) from the process description. "
        "These often start with 'if', 'whether', 'depending on', 'case'. Include all conditional paths originating from that single decision point. "
        "Format the output strictly as:\nExclusive gateway 1: <text span for gateway 1>\nExclusive gateway 2: <text span for gateway 2>\n... Output ONLY this structure."
    )
    user_msg = (
        # Примеры (можно проверить/дополнить)
        "Process: 'If the client opts for funding, complete a loan request then submit application. If the client pays cash, bring the full amount.'\n"
        "Exclusive gateway 1: If the client opts for funding, complete a loan request then submit application. If the client pays cash, bring the full amount.\n\n"
        "Process: 'Check inventory. If item is in stock, check payment. If item is out of stock, notify customer. Then, if payment authorized, confirm order. If payment declined, notify customer.'\n"
        "Exclusive gateway 1: If item is in stock, check payment. If item is out of stock, notify customer.\n"
        "Exclusive gateway 2: if payment authorized, confirm order. If payment declined, notify customer.\n\n"
        f"Process: '{process_description}'"
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    return _generate_llm_response(messages, max_new_tokens=512)


def extract_exclusive_gateways_2_conditions(process_description: str) -> str:
    """
    Извлекает текст для exclusive gateway с ровно двумя условиями.
    """
    system_msg = (
        "You are a BPMN expert. From the process description, extract the single text span covering an exclusive decision point (gateway) that has exactly **two** conditional paths (e.g., if/else, option A/option B). "
        "Include the complete text for both paths starting from the condition. "
        "Format the output strictly as:\nExclusive gateway 1: <text span containing both paths>"
    )
    user_msg = (
        # Примеры...
        "Process: 'If score < 60%, retake exam. Else (score >= 60%), enter grade.'\n"
        "Exclusive gateway 1: If score < 60%, retake exam. Else (score >= 60%), enter grade.\n\n"
        "Process: 'Check application. If complete, process it. If incomplete, request more info.'\n"
        "Exclusive gateway 1: If complete, process it. If incomplete, request more info.\n\n"
        f"Process: '{process_description}'"
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    return _generate_llm_response(messages, max_new_tokens=384)


# --- ФУНКЦИЯ УДАЛЕНА ---
# def extract_gateway_conditions(condition_list_str: str, gateway_list_str: str) -> str:
#     """
#     Группирует УЖЕ ИЗВЛЕЧЕННЫЕ условия по УЖЕ ИЗВЛЕЧЕННЫМ шлюзам.
#     (УДАЛЕНО - ненадежно, лучше делать в Python)
#     """
#     print(f"{Fore.YELLOW}Warning: LLM function 'extract_gateway_conditions' is deprecated and should not be called.{Fore.RESET}")
#     return "" # Возвращаем пустую строку, если все же будет вызвана
# --- КОНЕЦ УДАЛЕНИЯ ---


def extract_parallel_gateways(process_description: str) -> str:
    """
    Извлекает текст, относящийся к параллельным действиям (шлюзам).
    """
    system_prompt = (
        "You are a BPMN expert. Extract text spans describing activities performed **in parallel** or **concurrently**. "
        "Look for keywords like 'meanwhile', 'at the same time', 'in parallel', 'while' (used for concurrency), 'concurrently', 'simultaneously'. "
        "Include the text for all parallel branches belonging to the same concurrent execution block. "
        "Format the output strictly as:\nParallel gateway 1: <text span 1>\nParallel gateway 2: <text span 2>\n... Output ONLY this structure."
    )
    user_prompt = (
         # Примеры...
        "Process: 'Analyst evaluates credit. Meanwhile, another team does the same. After approval, team A verifies employment while team B verifies income simultaneously.'\n"
        "Parallel gateway 1: Analyst evaluates credit. Meanwhile, another team does the same.\n"
        "Parallel gateway 2: team A verifies employment while team B verifies income simultaneously.\n\n"
        "Process: 'Manager sends mail and prepares documents at the same time.'\n" # Пример с 'at the same time'
        "Parallel gateway 1: Manager sends mail and prepares documents at the same time.\n\n"
        f"Process: '{process_description}'"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return _generate_llm_response(messages, max_new_tokens=512)


def number_of_parallel_paths(parallel_gateway_text: str) -> str:
    """
    Определяет количество параллельных путей в описании шлюза.
    """
    system_prompt = (
        "Analyze the provided text describing parallel activities. Determine the number of distinct parallel paths/branches described. "
        "Respond with a single digit only (e.g., '2', '3')."
    )
    user_prompt = (
        # Примеры...
        "Text: 'Team A does X. Team B does Y. Team C does Z simultaneously.'\n"
        "Number: 3\n\n"
        "Text: 'Evaluates creditworthiness and collateral. Meanwhile, another team does the same.'\n"
        "Number: 2\n\n"
        "Text: 'Sends email while preparing the report.'\n"
        "Number: 2\n\n"
        f"Text: '{parallel_gateway_text}'\n"
        "Number:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = _generate_llm_response(messages, max_new_tokens=5)
    # Валидация ответа - должен быть одной цифрой
    if response.isdigit() and len(response) == 1:
        return response
    else:
        print(f"{Fore.YELLOW}Warning: LLM returned non-digit for number_of_parallel_paths: '{response}'. Defaulting to 2.{Fore.RESET}")
        return "2" # Возвращаем 2 по умолчанию при некорректном ответе


def extract_parallel_tasks(sentence: str) -> str:
    """
    Извлекает параллельные задачи из ОДНОГО предложения.
    """
    system_prompt = (
        "You are an expert in structuring process sentences. Given a single sentence describing multiple tasks done concurrently, "
        "extract each distinct task action. "
        "Format the output strictly as:\nTask 1: <task description 1>\nTask 2: <task description 2>\n... Output ONLY this structure."
    )
    user_prompt = (
        # Примеры...
        'Sentence: "The chef simultaneously prepares the entree and makes the salad."\n'
        'Task 1: prepares the entree\n'
        'Task 2: makes the salad\n\n'
        'Sentence: "Manager coordinates with design, development, and QA teams in parallel."\n'
        'Task 1: coordinates with design team\n'
        'Task 2: coordinates with development team\n'
        'Task 3: coordinates with QA team\n\n'
        f'Sentence: "{sentence}"'
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return _generate_llm_response(messages, max_new_tokens=128)


def extract_3_parallel_paths(parallel_gateway_text: str) -> str:
    """
    Извлекает 3 параллельных пути из текста шлюза.
    """
    system_prompt = (
        "You are a BPMN expert. From the provided text describing concurrent activities, extract exactly **three** distinct parallel paths. "
        "Each path might contain multiple actions. "
        "Separate the paths strictly using ' && ' (space, ampersands, space). Use ' && ' exactly twice. Output ONLY the paths in this format."
    )
    user_prompt = (
         # Примеры...
        "Text: 'Path A: task A1, then A2. Path B: task B1. Path C: task C1, C2.'\n"
        "Paths: Path A: task A1, then A2 && Path B: task B1 && Path C: task C1, C2\n\n"
        "Text: 'Kitchen team analyzes ideas and creates recipe. Simultaneously, customer service does research while art team creates concepts. Concurrently, accountants review cost.'\n"
        "Paths: Kitchen team analyzes ideas and creates recipe && customer service does research while art team creates concepts && accountants review cost\n\n"
        f"Text: {parallel_gateway_text}\n"
        "Paths:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = _generate_llm_response(messages, max_new_tokens=384)
    if response.count("&&") != 2:
        print(f"{Fore.YELLOW}Warning: LLM response for 3 paths had {response.count('&&')} separators instead of 2.{Fore.RESET}")
    return response


def extract_2_parallel_paths(parallel_gateway_text: str) -> str:
    """
    Извлекает 2 параллельных пути из текста шлюза.
    """
    system_prompt = (
        "You are a BPMN expert. From the provided text describing concurrent activities, extract exactly **two** distinct parallel paths. "
        "Each path might contain multiple actions. "
        "Separate the paths strictly using ' && ' (space, ampersands, space). Use ' && ' exactly once. Output ONLY the paths in this format."
    )
    user_prompt = (
         # Примеры...
        "Text: 'He delivers mail and greets people. Simultaneously, the milkman delivers milk.'\n"
        "Paths: He delivers mail and greets people && the milkman delivers milk\n\n"
        "Text: 'Team A revises the design while Team B implements approved parts.'\n"
        "Paths: Team A revises the design && Team B implements approved parts\n\n"
        f"Text: {parallel_gateway_text}\n"
        "Paths:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = _generate_llm_response(messages, max_new_tokens=384)
    if response.count("&&") != 1:
         print(f"{Fore.YELLOW}Warning: LLM response for 2 paths had {response.count('&&')} separators instead of 1.{Fore.RESET}")
    return response


def find_previous_task(task_word: str, previous_tasks_str: str) -> str:
    """
    Определяет наиболее вероятную предшествующую задачу для цикла.
    """
    system_prompt = (
        "You are a business process analyst. A task repeats (e.g., contains 'again', 'repeat'). You are given this repeating task and a list of previous tasks with IDs (e.g., T0: text). "
        "Identify which previous task the repeating task should loop back to, considering the process flow. "
        "Output ONLY the exact text of the selected previous task from the provided list."
    )
    user_prompt = (
        # Примеры...
        "Repeating Task: 'discusses contract again with sales'\n"
        "Previous Tasks:\n"
        "T0: decide payment method\n"
        "T1: fill loan application\n"
        "T5: sign the contract\n" # <-- Target
        "Selected Previous Task: sign the contract\n\n"

        "Repeating Task: 'takes the exam again'\n"
        "Previous Tasks:\n"
        "T0: take the exam\n" # <-- Target
        "T1: professor enters grade\n"
        "Selected Previous Task: take the exam\n\n" # Предполагаем, что задача 'take the exam' была ранее

        f"Repeating Task: '{task_word}'\n"
        f"Previous Tasks:\n{previous_tasks_str}\n"
        "Selected Previous Task:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return _generate_llm_response(messages, max_new_tokens=64)

def determine_gateway_flow(gateway_id: str, conditions: list[str], subsequent_tasks: list[dict], context_text: str = "") -> str:
    """
    Использует LLM для определения начальных задач веток и точки слияния после шлюза.

    Args:
        gateway_id: ID расходящегося шлюза (например, EG0).
        conditions: Список текстов условий для каждой ветки.
        subsequent_tasks: Список словарей задач, следующих за шлюзом,
                          каждый вида {"id": "T1", "text": "fill out form"}.
        context_text: Опциональный контекстный текст процесса.

    Returns:
        Строка JSON с описанием начальных точек и слияния или пустая строка при ошибке.
        Формат JSON:
        {
          "gateway_id": "...",
          "branches": [
            {"condition_index": 0, "immediate_task_id": "..."},
            {"condition_index": 1, "immediate_task_id": "..."}
          ],
          "merge_task_id": "..." | null
        }
    """
    if not model_loaded or (not conditions and gateway_id.startswith("EG")) or not subsequent_tasks: # Условиям нужен только XOR шлюз
        print(f"{Fore.YELLOW}Warning: Missing prerequisites for LLM gateway flow call (model loaded: {model_loaded}, conditions: {bool(conditions)}, tasks: {bool(subsequent_tasks)}){Fore.RESET}")
        return ""

    system_prompt = (
        "You are a BPMN expert analyzing process flows after a decision or parallel execution point (gateway).\n"
        "You will be given the gateway ID, its conditions (if applicable), and a list of subsequent tasks with their IDs.\n"
        "Your goal is to determine the IMMEDIATE task following each condition (or starting each parallel path) and the task where these paths MERGE, if any.\n"
        "Output ONLY a valid JSON object adhering strictly to the specified format. Do not include any explanations or introductory text."
    )

    # Форматируем задачи и условия
    task_list_str = "\n".join([f"- {task['id']}: {task['text']}" for task in subsequent_tasks])
    condition_list_str = ""
    if conditions: # Добавляем условия только если они есть (для XOR)
        condition_list_str = "\n".join([f"- Condition {chr(65+i)}: {cond}" for i, cond in enumerate(conditions)])
    else: # Для параллельных шлюзов
         condition_list_str = "- N/A (Parallel Gateway)"


    user_prompt = (
        f"Gateway ID: {gateway_id}\n\n"
        f"Conditions/Paths:\n{condition_list_str}\n\n"
        f"Subsequent Tasks:\n{task_list_str}\n\n"
        f"Context Text (optional):\n{context_text[:1000]}\n\n" # Ограничиваем длину контекста
        "Determine the flow structure. "
        "Identify the immediate task ID for each condition/path (A, B, ... or Path 1, Path 2 for parallel) and the merge task ID (use the task ID from the list, or null if no merge).\n"
        "Output JSON format:\n"
        "{\n"
        '  "gateway_id": "...",\n'
        '  "branches": [\n'
        '    {"condition_index": 0, "immediate_task_id": "..."},\n' # Убрали sequence
        '    {"condition_index": 1, "immediate_task_id": "..."}\n'
        '  ],\n'
        '  "merge_task_id": "..." | null\n'
        "}\n\n"
        "JSON Output:"
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = _generate_llm_response(messages, max_new_tokens=256, temperature=0.1) # Снизим max_tokens и температуру

    # Попытка извлечь JSON
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                 # Базовая проверка ключей перед полным парсингом
                 temp_data = json.loads(json_str)
                 if "gateway_id" in temp_data and "branches" in temp_data and "merge_task_id" in temp_data:
                     return json_str
                 else:
                      print(f"{Fore.YELLOW}Warning: LLM JSON response for gateway flow missing required keys:\n{json_str}{Fore.RESET}")
                      return ""
            except json.JSONDecodeError:
                 print(f"{Fore.YELLOW}Warning: LLM response for gateway flow is not valid JSON:\n{response}{Fore.RESET}")
                 return ""
        else:
            print(f"{Fore.YELLOW}Warning: Could not extract JSON from LLM response for gateway flow:\n{response}{Fore.RESET}")
            return ""
    except Exception as e:
        print(f"{Fore.RED}Error processing LLM response for gateway flow: {e}{Fore.RESET}")
        return ""


# --- END: Добавление новой функции в deepseek_prompts.py ---