# --- START OF FILE create_bpmn_structure.py ---

from logging_utils import write_to_file

# TODO: Адаптировать класс GraphGenerator для использования новой структуры агента!
#       Имя агента для дорожки теперь находится в element['content']['agent']['resolved_word']

def create_bpmn_structure(
    agent_task_pairs: list[dict],
    parallel_gateway_data: list[dict],
    exclusive_gateway_data: list[dict],
    process_info: list[dict],
) -> list[dict] | None: # Добавлен None для возможной ошибки
    """
    Creates a BPMN structure from the agent-task pairs, parallel gateways and exclusive gateways.
    The BPMN structure can be used to create a visual representation of the BPMN diagram.

    Args:
        agent_task_pairs (list[dict]): A list of agent-task pairs.
                                       Expected structure for agent: pair['agent'] =
                                       {'original_word': str, 'resolved_word': str, 'entity': dict}
                                       OR pair['agent'] might be missing if it's a loop element.
        parallel_gateway_data (list[dict]): A list of parallel gateway data.
        exclusive_gateway_data (list[dict]): A list of exclusive gateway data.
        process_info (list[dict]): A list of process info entities.

    Returns:
        list[dict] | None: A list of BPMN structure elements ready for GraphGenerator, or None if input is invalid.
    """
    if not isinstance(agent_task_pairs, list):
         print(f"ERROR in create_bpmn_structure: Invalid input 'agent_task_pairs' (expected list, got {type(agent_task_pairs)}).")
         return None

    # Шаг 1: Форматирование пар (перенос данных в 'content')
    # Эта функция корректно обработает новую структуру агента внутри content.
    formatted_pairs = format_agent_task_pairs(agent_task_pairs)

    # Копируем для дальнейшей модификации (добавление в шлюзы)
    elements_to_process = formatted_pairs.copy()

    # Шаг 2: Объединение и сортировка шлюзов
    gateways = parallel_gateway_data + exclusive_gateway_data
    # Сортируем по "размеру" (разница между концом и началом) - не уверен, лучшая ли это сортировка?
    # Возможно, лучше сортировать по началу: sorted(gateways, key=lambda x: x.get('start', float('inf')))
    gateways = sorted(gateways, key=calculate_distance) # Используем существующую функцию

    # Шаг 3: Добавление задач/циклов в соответствующие пути шлюзов
    # Эта функция модифицирует и gateways (добавляя children), и elements_to_process (удаляя добавленные)
    add_tasks_to_gateways(elements_to_process, gateways, process_info)
    write_to_file("bpmn_structure/gateways_with_children.json", gateways) # Переименовал лог

    # Шаг 4: Построение иерархии вложенных шлюзов
    # Эта функция работает с уже модифицированным списком gateways
    nested_gateways = nest_gateways(gateways)
    write_to_file("bpmn_structure/nested_gateways_structure.json", nested_gateways) # Переименовал лог

    # Шаг 5: Объединение оставшихся задач/циклов и шлюзов верхнего уровня
    # elements_to_process теперь содержит только те элементы, что не попали ни в один шлюз
    final_structure = elements_to_process + nested_gateways

    # Шаг 6: Финальная сортировка по начальной позиции
    final_structure = sorted(final_structure, key=lambda x: get_start_idx(x) if get_start_idx(x) is not None else float('inf'))

    print("Final BPMN structure created.")
    # Запись финальной структуры в файл (как и раньше)
    write_to_file("bpmn_structure/bpmn_final_structure.json", final_structure) # Переименовал лог

    return final_structure


def get_start_idx(element: dict) -> int | None:
    """
    Safely gets the start index of a BPMN structure element.
    Args:
        element (dict): A BPMN structure element (task, loop, gateway).
    Returns:
        int | None: The start index or None if not found.
    """
    if isinstance(element, dict):
        # Для шлюзов или элементов цикла с ключом 'start'
        if "start" in element:
            return element["start"]
        # Для отформатированных пар (задач)
        elif "content" in element and isinstance(element["content"], dict):
            content = element["content"]
            # Если это задача
            if "task" in content and isinstance(content["task"], dict):
                return content["task"].get("start") # Используем .get
            # Если это был цикл (go_to), у него должен быть 'start' напрямую в content
            elif "start" in content:
                 return content["start"]
    # Если не удалось найти индекс
    return None


def format_agent_task_pairs(agent_task_pairs: list[dict]) -> list[dict]:
    """
    Formats agent-task pairs into the structure expected by subsequent functions.
    Moves original pair content into a 'content' key and adds a 'type' key.
    Args:
        agent_task_pairs (list[dict]): The list of agent-task pairs from process_bpmn_data.
                                       Handles new agent structure: pair['agent'] = {'resolved_word': ...}
    Returns:
        list[dict]: Formatted list.
    """
    formatted_list = []
    for original_pair in agent_task_pairs:
        # Создаем новый словарь для форматированного элемента
        formatted_pair = {}
        # Копируем все содержимое исходной пары в ключ 'content'
        # Это автоматически включает новую структуру 'agent': {'original_word': ..., 'resolved_word': ..., 'entity': ...}
        formatted_pair["content"] = original_pair.copy()
        # Определяем тип элемента
        if "task" in original_pair:
            formatted_pair["type"] = "task"
        elif "go_to" in original_pair: # Проверяем наличие ключа цикла
            formatted_pair["type"] = "loop"
            # Добавляем start/end из content в корень для сортировки/привязки, если их нет
            if "start" not in formatted_pair and "start" in formatted_pair["content"]:
                 formatted_pair["start"] = formatted_pair["content"]["start"]
            if "end" not in formatted_pair and "end" in formatted_pair["content"]:
                 formatted_pair["end"] = formatted_pair["content"]["end"]
        else:
            formatted_pair["type"] = "unknown" # На случай непредвиденной структуры
            print(f"Warning in format_agent_task_pairs: Unknown pair type: {original_pair}")

        formatted_list.append(formatted_pair)

    return formatted_list


def gateway_contains_nested_gateways(gateway: dict, all_gateways: list[dict]) -> bool:
    """
    Checks if a gateway contains any other gateways strictly within its start/end bounds.
    Args:
        gateway (dict): The potential outer gateway.
        all_gateways (list[dict]): The list of all gateways.
    Returns:
        bool: True if it contains nested gateways, False otherwise.
    """
    gw_start = gateway.get("start")
    gw_end = gateway.get("end")

    if gw_start is None or gw_end is None:
        return False # Не можем проверить без границ

    for other_g in all_gateways:
        # Пропускаем сравнение шлюза с самим собой
        if other_g.get("id") == gateway.get("id"):
            continue

        other_start = other_g.get("start")
        other_end = other_g.get("end")

        if other_start is not None and other_end is not None:
            # Строгое вложение: другой шлюз начинается после начала текущего
            # и заканчивается до конца текущего.
            if other_start > gw_start and other_end < gw_end:
                return True
    return False


def add_tasks_to_gateways(
    elements_to_process: list[dict], gateways: list[dict], process_info: list[dict]
) -> None:
    """
    Adds BPMN elements (tasks, loops) to the corresponding gateway paths based on their start indices.
    Modifies 'gateways' by adding 'children' lists and 'elements_to_process' by removing added elements.
    Args:
        elements_to_process (list[dict]): List of formatted BPMN elements (tasks/loops) to potentially add.
        gateways (list[dict]): List of gateway dictionaries (will be modified).
        process_info (list[dict]): List of process info entities (used for handling PROCESS_CONTINUE).
    Returns:
        None
    """
    processed_element_indices = set() # Отслеживаем добавленные элементы по индексу в исходном списке

    for gateway in gateways:
        gateway_id = gateway.get("id", "unknown_gateway")
        gateway_start = gateway.get("start")
        gateway_end = gateway.get("end")
        gateway_paths = gateway.get("paths", [])

        # Инициализируем 'children', если их нет
        if "children" not in gateway:
             gateway["children"] = [[] for _ in range(len(gateway_paths))]
        # Убедимся, что количество списков children совпадает с количеством путей
        elif len(gateway["children"]) != len(gateway_paths):
             print(f"Warning: Mismatch between children lists ({len(gateway['children'])}) and paths ({len(gateway_paths)}) for gateway {gateway_id}. Reinitializing children.")
             gateway["children"] = [[] for _ in range(len(gateway_paths))]


        gateway["type"] = "parallel" if gateway_id.startswith("PG") else "exclusive"

        for i, path in enumerate(gateway_paths):
            path_start = path.get("start")
            path_end = path.get("end")

            if path_start is None or path_end is None:
                print(f"Warning: Path {i} in gateway {gateway_id} has invalid indices. Skipping.")
                continue

            # Ищем элементы, попадающие в текущий путь
            elements_in_path = []
            indices_to_remove = [] # Индексы для удаления из elements_to_process

            for elem_idx, element in enumerate(elements_to_process):
                 if elem_idx in processed_element_indices:
                      continue # Пропускаем уже обработанные

                 elem_start_idx = get_start_idx(element)

                 if elem_start_idx is not None:
                     # Элемент принадлежит пути, если его начало внутри диапазона пути
                     # Используем path_end + 1, т.к. range не включает верхнюю границу
                     if path_start <= elem_start_idx < path_end + 1:
                         elements_in_path.append(element)
                         indices_to_remove.append(elem_idx) # Помечаем для удаления

            # Добавляем найденные элементы в children шлюза
            if elements_in_path:
                 # Сортируем элементы внутри пути по их началу
                 elements_in_path.sort(key=lambda x: get_start_idx(x) if get_start_idx(x) is not None else float('inf'))
                 gateway["children"][i].extend(elements_in_path)
                 # Обновляем множество обработанных индексов
                 processed_element_indices.update(indices_to_remove)

            # --- Обработка условий (перенесена) ---
            # Удаляем 'condition' из всех элементов пути, кроме первого (для эксклюзивных)
            # Или сохраняем первое условие на уровне шлюза (для параллельных - ???)
            children_list = gateway["children"][i]
            first_condition_found = None
            for child_idx, child in enumerate(children_list):
                 if "condition" in child.get("content", {}):
                     if gateway["type"] == "exclusive":
                          if child_idx > 0: # Удаляем у всех, кроме первого
                              del child["content"]["condition"]
                     elif gateway["type"] == "parallel":
                           # В параллельном шлюзе условие обычно относится ко всему шлюзу
                           if first_condition_found is None:
                                first_condition_found = child["content"]["condition"]
                           # Удаляем из всех дочерних элементов в параллельном пути
                           del child["content"]["condition"]

            # Сохраняем первое найденное условие на уровне параллельного шлюза (если нужно)
            if gateway["type"] == "parallel" and first_condition_found and "condition" not in gateway:
                 gateway["condition"] = first_condition_found
            # --- Конец обработки условий ---


        # --- Обработка PROCESS_CONTINUE (после добавления всех задач) ---
        if gateway["type"] == "exclusive":
            # Ищем PROCESS_CONTINUE сущности, связанные с этим шлюзом
            process_continue_entities = [
                e
                for e in process_info
                if e["entity_group"] == "PROCESS_CONTINUE"
                and gateway_start is not None and gateway_end is not None # Проверка границ шлюза
                and gateway_start <= e.get("start", -1) < gateway_end + 1
            ]

            # Если нашли ровно одну сущность "продолжение" для этого шлюза
            if len(process_continue_entities) == 1:
                # И если шлюз НЕ содержит вложенных шлюзов (логика handle_process_continue)
                if not gateway_contains_nested_gateways(gateway, gateways):
                     handle_process_continue_entity(
                         elements_to_process, # Передаем ОСТАВШИЕСЯ элементы
                         gateways, # Передаем все шлюзы
                         gateway # Текущий шлюз
                     )
            elif len(process_continue_entities) > 1:
                 print(f"Warning: Multiple PROCESS_CONTINUE entities found within exclusive gateway {gateway_id}. Behavior undefined.")
        # --- Конец обработки PROCESS_CONTINUE ---


    # Удаляем элементы, которые были добавлены в шлюзы, из основного списка
    # Идем по индексам в обратном порядке, чтобы не сбить нумерацию при удалении
    if processed_element_indices:
         sorted_indices_to_remove = sorted(list(processed_element_indices), reverse=True)
         for index in sorted_indices_to_remove:
              if 0 <= index < len(elements_to_process): # Доп. проверка индекса
                  del elements_to_process[index]
              else:
                   print(f"Warning: Attempted to remove element at invalid index {index}.")


def handle_process_continue_entity(
    remaining_elements: list[dict], all_gateways: list[dict], current_gateway: dict
) -> None:
    """
    Handles the PROCESS_CONTINUE entity in exclusive gateways by adding a "continue" element
    to empty paths, pointing to the next element after the gateway.
    Modifies the 'children' list of the current_gateway.
    Args:
        remaining_elements (list[dict]): List of BPMN elements *not* assigned to any gateway yet.
        all_gateways (list[dict]): List of all gateways (used for context, not modified).
        current_gateway (dict): The exclusive gateway dictionary (will be modified).
    Returns:
        None
    """
    # Эта логика применяется только если шлюз НЕ содержит вложенных (уже проверено снаружи)
    gateway_end_idx = current_gateway.get("end")
    if gateway_end_idx is None:
        return # Не можем определить следующий элемент

    next_element_id = None
    min_start_after_gateway = float('inf')

    # Ищем первый элемент (задачу или цикл), который начинается СТРОГО после текущего шлюза
    for element in remaining_elements:
        elem_start = get_start_idx(element)
        if elem_start is not None and elem_start > gateway_end_idx:
            if elem_start < min_start_after_gateway:
                 min_start_after_gateway = elem_start
                 # Получаем ID задачи, если это задача
                 if element.get("type") == "task" and "task_id" in element.get("content", {}).get("task", {}):
                      next_element_id = element["content"]["task"]["task_id"]
                 # Если это цикл, у него нет ID, но он может быть следующим элементом
                 # В BPMN обычно 'continue' ведет к задаче или merge-шлюзу.
                 # Пока просто запоминаем, что есть следующий элемент.
                 elif element.get("type") == "loop":
                      # Может быть, стоит искать следующую *задачу*, а не цикл?
                      pass # Игнорируем цикл как цель для continue?

    # Если не нашли следующий элемент среди оставшихся, ищем в других шлюзах
    if next_element_id is None:
         next_gateway = None
         min_gw_start_after = float('inf')
         for gw in all_gateways:
             gw_start = gw.get("start")
             # Ищем шлюз, начинающийся строго после текущего
             if gw_start is not None and gw_start > gateway_end_idx:
                  if gw_start < min_gw_start_after:
                       min_gw_start_after = gw_start
                       next_gateway = gw
         if next_gateway:
              # Цель - сам ID следующего шлюза
              next_element_id = next_gateway.get("id")

    # Если нашли куда переходить (ID задачи или шлюза)
    if next_element_id:
        # Добавляем элемент 'continue' во все ПУСТЫЕ пути текущего шлюза
        for i in range(len(current_gateway.get("children", []))):
            if not current_gateway["children"][i]: # Если путь пуст
                current_gateway["children"][i].append(
                    {"content": {"go_to": next_element_id}, "type": "continue"}
                )
                print(f"Added 'continue' element to path {i} of gateway {current_gateway.get('id')}, pointing to {next_element_id}")
    else:
         print(f"Warning: Could not find next element after exclusive gateway {current_gateway.get('id')} to point 'continue' elements to.")


def calculate_distance(gateway: dict) -> int:
    """
    Calculates the 'distance' or span of a gateway (end - start).
    Used for sorting gateways. Returns a large number if indices are missing.
    Args:
        gateway (dict): Gateway dictionary.
    Returns:
        int: The difference between end and start, or infinity if indices are invalid.
    """
    start = gateway.get("start")
    end = gateway.get("end")
    if start is not None and end is not None and end >= start:
        return end - start
    else:
        # Возвращаем большое значение, чтобы некорректные шлюзы оказались в конце сортировки
        return float('inf')


def nest_gateways(all_gateways: list[dict]) -> list[dict]:
    """
    Nests gateways based on their start/end indices and parent references (if added previously).
    Args:
        all_gateways (list[dict]): A list of gateway dictionaries (potentially modified by add_tasks_to_gateways).
    Returns:
        list[dict]: A list of top-level gateways, with nested gateways moved inside their parents' 'children'.
    """
    gateway_map = {gw.get("id"): gw for gw in all_gateways if gw.get("id")}
    nested_ids = set() # Сохраняем ID шлюзов, которые были вложены

    # Сначала обрабатываем явные ссылки родитель-потомок (из extract_exclusive_gateways)
    for gateway in all_gateways:
        parent_id = gateway.get("parent_gateway_id")
        parent_path_idx = gateway.get("parent_gateway_path_id") # Используем индекс пути

        if parent_id and parent_id in gateway_map and parent_path_idx is not None:
            parent_gw = gateway_map[parent_id]
            # Убедимся, что у родителя есть 'children' и нужный индекс пути
            if "children" in parent_gw and 0 <= parent_path_idx < len(parent_gw["children"]):
                # Ищем, куда вставить вложенный шлюз, сохраняя порядок по 'start'
                target_path_children = parent_gw["children"][parent_path_idx]
                insert_in_sorted_order(target_path_children, gateway)
                nested_ids.add(gateway.get("id")) # Помечаем как вложенный
            else:
                 print(f"Warning: Could not nest gateway {gateway.get('id')} into parent {parent_id} path {parent_path_idx}. Parent structure invalid.")

    # Затем обрабатываем вложенность на основе координат для тех, у кого нет явного родителя
    # Идем в обратном порядке (от меньших к большим), чтобы вложить внутренние первыми
    sorted_gateways = sorted(all_gateways, key=calculate_distance)

    for i, inner_gw in enumerate(sorted_gateways):
        inner_id = inner_gw.get("id")
        # Пропускаем уже вложенные по явной ссылке
        if inner_id in nested_ids:
            continue

        best_parent = None
        best_parent_path_idx = -1
        min_parent_distance = float('inf')

        # Ищем наименьший подходящий родительский шлюз среди оставшихся
        for j, outer_gw in enumerate(sorted_gateways):
            outer_id = outer_gw.get("id")
            # Не вкладываем в себя и не вкладываем в уже вложенные шлюзы
            if i == j or outer_id in nested_ids:
                continue

            outer_paths = outer_gw.get("paths", [])
            outer_children = outer_gw.get("children") # Нужны children для вставки

            # Проверяем вложение по координатам и размеру
            # is_nested проверяет start/end шлюза, а не пути!
            if is_nested(inner_gw, outer_gw):
                 current_distance = calculate_distance(outer_gw)
                 # Ищем самый "маленький" (наиболее близкий по размеру) родитель
                 if current_distance < min_parent_distance:
                     # Теперь ищем, в какой *путь* родителя вложить
                     path_found = False
                     for path_idx, path in enumerate(outer_paths):
                         # Проверяем, попадает ли начало внутреннего шлюза в путь родителя
                         if path.get("start") is not None and path.get("end") is not None and \
                            path["start"] <= inner_gw.get("start", -1) < path["end"] + 1:
                             best_parent = outer_gw
                             best_parent_path_idx = path_idx
                             min_parent_distance = current_distance
                             path_found = True
                             break # Нашли подходящий путь
                     # if not path_found: # Не нашли путь, куда вложить (странно)
                     #     print(f"Warning: Gateway {inner_id} seems nested in {outer_id}, but not within any specific path.")

        # Если нашли родителя
        if best_parent and best_parent_path_idx != -1:
            # Убедимся, что у родителя есть 'children' и нужный индекс пути
            if "children" in best_parent and 0 <= best_parent_path_idx < len(best_parent["children"]):
                target_path_children = best_parent["children"][best_parent_path_idx]
                insert_in_sorted_order(target_path_children, inner_gw)
                nested_ids.add(inner_id) # Помечаем как вложенный
            else:
                 print(f"Warning: Could not nest gateway {inner_id} into parent {best_parent.get('id')} path {best_parent_path_idx} based on coordinates. Parent structure invalid.")


    # Возвращаем только шлюзы верхнего уровня (те, что не были вложены)
    top_level_gateways = [gw for gw in all_gateways if gw.get("id") not in nested_ids]

    return top_level_gateways


def insert_in_sorted_order(children: list, element_to_insert: dict):
    """
    Inserts an element into a list of children, maintaining sort order by start index.
    Args:
        children (list): The list of child elements (tasks, loops, gateways).
        element_to_insert (dict): The element to insert.
    """
    insert_start_idx = get_start_idx(element_to_insert)
    if insert_start_idx is None:
        # Если не можем определить начало, добавляем в конец
        children.append(element_to_insert)
        return

    index = 0
    while index < len(children):
        child_start_idx = get_start_idx(children[index])
        # Если у ребенка нет индекса или начало вставляемого элемента меньше
        if child_start_idx is None or insert_start_idx < child_start_idx:
            break # Нашли позицию для вставки
        index += 1
    children.insert(index, element_to_insert)


# --- Функция ranges_overlap_percentage БЕЗ ИЗМЕНЕНИЙ ---
def ranges_overlap_percentage(
    range1: tuple[int, int], range2: tuple[int, int], min_overlap_percentage=0.97
) -> bool:
    """
    Determines if two ranges overlap by a certain percentage. Each range is a tuple of the form (start, end).
    Args:
        range1 (tuple): The first range.
        range2 (tuple): The second range.
        min_overlap_percentage (float): The minimum percentage of overlap required for the ranges to be considered overlapping.
    Returns:
        bool: True if the ranges overlap by the minimum percentage, False otherwise.
    """
    start1, end1 = range1
    start2, end2 = range2

    # Проверка на валидность диапазонов
    if start1 >= end1 or start2 >= end2:
        return False # Диапазон нулевой или отрицательной длины

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    # Есть ли пересечение?
    if overlap_start < overlap_end:
        overlap_range = overlap_end - overlap_start
        range1_size = end1 - start1
        range2_size = end2 - start2

        # Избегаем деления на ноль (хотя уже проверили start < end)
        if range1_size == 0 or range2_size == 0:
             return False

        overlap_percentage1 = overlap_range / range1_size
        overlap_percentage2 = overlap_range / range2_size

        # Проверяем, что ОБА процента перекрытия удовлетворяют порогу
        return (
            overlap_percentage1 >= min_overlap_percentage
            and overlap_percentage2 >= min_overlap_percentage
        )
    else:
        # Нет пересечения
        return False


# --- Блок __main__ остается для тестирования ---
# Он будет читать JSON с новой структурой agent_task_pairs,
# и код выше должен его корректно обработать.
if __name__ == "__main__":
    import json
    from os.path import exists

    # Определяем пути к файлам логов
    log_dir = "output_logs"
    structure_dir = "bpmn_structure"
    atp_final_file = f"{log_dir}/agent_task_pairs_final.json"
    proc_info_file = f"{log_dir}/process_info_entities_classified.json" # Используем классифицированные
    pg_data_file = f"{log_dir}/parallel_gateway_data.json"
    eg_data_file = f"{log_dir}/exclusive_gateway_data.json"
    output_file = f"{structure_dir}/bpmn_final_structure.json" # Имя финального файла

    # Инициализируем переменные
    agent_task_pairs = []
    parallel_gateway_data = []
    exclusive_gateway_data = []
    process_info = []

    # Загружаем данные из файлов логов, если они существуют
    if exists(atp_final_file):
        try:
            with open(atp_final_file, "r", encoding='utf-8') as file: # Добавил encoding
                agent_task_pairs = json.load(file)
            print(f"Loaded {len(agent_task_pairs)} items from {atp_final_file}")
        except Exception as e:
            print(f"Error loading {atp_final_file}: {e}")
    else:
        print(f"File not found: {atp_final_file}")


    if exists(proc_info_file):
         try:
             with open(proc_info_file, "r", encoding='utf-8') as file:
                 process_info = json.load(file)
             print(f"Loaded {len(process_info)} items from {proc_info_file}")
         except Exception as e:
             print(f"Error loading {proc_info_file}: {e}")
    # else: # Отсутствие process_info может быть нормальным
    #     print(f"File not found: {proc_info_file}")


    if exists(pg_data_file):
        try:
            with open(pg_data_file, "r", encoding='utf-8') as file:
                parallel_gateway_data = json.load(file)
            print(f"Loaded {len(parallel_gateway_data)} items from {pg_data_file}")
        except Exception as e:
            print(f"Error loading {pg_data_file}: {e}")
    # else: # Отсутствие параллельных шлюзов - норма
    #     print(f"File not found: {pg_data_file}")


    if exists(eg_data_file):
        try:
            with open(eg_data_file, "r", encoding='utf-8') as file:
                exclusive_gateway_data = json.load(file)
            print(f"Loaded {len(exclusive_gateway_data)} items from {eg_data_file}")
        except Exception as e:
            print(f"Error loading {eg_data_file}: {e}")
    # else: # Отсутствие эксклюзивных шлюзов - норма
    #     print(f"File not found: {eg_data_file}")

    # Проверяем, есть ли хотя бы пары агент-задача для обработки
    if not agent_task_pairs and not parallel_gateway_data and not exclusive_gateway_data:
         print("No input data found to create BPMN structure. Exiting.")
    else:
        print("\nRunning create_bpmn_structure...")
        # Вызываем основную функцию
        final_structure = create_bpmn_structure(
            agent_task_pairs, parallel_gateway_data, exclusive_gateway_data, process_info
        )

        # Проверяем результат и записываем в файл
        if final_structure is not None:
            # Создаем директорию для структуры, если ее нет
            import os
            if not os.path.exists(structure_dir):
                os.makedirs(structure_dir)
            # Записываем финальную структуру
            write_to_file(output_file, final_structure)
            print(f"\nFinal BPMN structure saved to {output_file}")
        else:
            print("\nFailed to create BPMN structure.")