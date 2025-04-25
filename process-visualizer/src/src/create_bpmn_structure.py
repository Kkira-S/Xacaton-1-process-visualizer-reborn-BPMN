# --- START OF FILE create_bpmn_structure.py ---

import uuid
from collections import defaultdict
import math
from colorama import Fore, init # Добавлено для цветного вывода
import json # <-- ДОБАВИТЬ ИМПОРТ JSON
import deepseek_prompts as prompts # <-- Убедитесь, что импорт есть
import traceback # <-- Для отладки

init(autoreset=True) # Инициализация colorama

# --- Константы для раскладки (можно настроить) ---
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 80
EVENT_WIDTH = 36
EVENT_HEIGHT = 36
GATEWAY_WIDTH = 50
GATEWAY_HEIGHT = 50
POOL_PADDING_X = 50
POOL_PADDING_Y = 50
LANE_HEADER_WIDTH = 30
LANE_PADDING_Y = 40 # Увеличим отступ внутри дорожки
VERTICAL_SPACING = 100 # Увеличим вертикальный отступ
HORIZONTAL_SPACING = 200 # Увеличим горизонтальный отступ
LANE_MIN_HEIGHT = DEFAULT_HEIGHT + LANE_PADDING_Y * 2
CONNECTION_POINT_OFFSET = 5

def generate_unique_id(prefix=""):
    """Генерирует уникальный ID с префиксом."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_element_center(bounds):
    """Возвращает координаты центра элемента."""
    if not bounds or not all(k in bounds for k in ['x', 'y', 'width', 'height']):
        return {"x": 0, "y": 0}
    return {
        "x": bounds["x"] + bounds["width"] // 2,
        "y": bounds["y"] + bounds["height"] // 2
    }

def find_element_by_internal_id(elements_data: list, internal_id: str) -> dict | None:
    """Находит элемент в списке all_elements_data по internal_id."""
    if not internal_id: return None # Проверка на пустой ID
    for elem in elements_data:
        if elem.get("internal_id") == internal_id:
            return elem
    return None

def find_first_element_after(elements_data: list, start_search_index: int, element_types: list[str]) -> dict | None:
    """Находит первый элемент указанных типов после заданного индекса."""
    if start_search_index < 0 or start_search_index >= len(elements_data) -1 :
         return None
    for i in range(start_search_index + 1, len(elements_data)):
        elem_type = elements_data[i].get("type")
        if elem_type in element_types:
            return elements_data[i]
    return None

def find_element_by_bpmn_id(flow_elements: list, bpmn_id: str) -> dict | None:
     """Находит семантический элемент по его BPMN ID."""
     return next((elem for elem in flow_elements if elem.get("id") == bpmn_id), None)


def create_bpmn_structure(
    agent_task_pairs_or_loops: list, # Изменено имя параметра
    parallel_gateways: list,
    exclusive_gateways: list,
    process_info_entities: list,
    unique_agents: list,
    original_text: str
) -> dict | None:
    print("Starting BPMN JSON structure creation...")
    if not isinstance(agent_task_pairs_or_loops, list) or \
       not isinstance(parallel_gateways, list) or \
       not isinstance(exclusive_gateways, list) or \
       not isinstance(unique_agents, list):
        print(f"{Fore.RED}Error: Invalid input data types for structure creation.{Fore.RESET}")
        return None

    # --- 1. Инициализация ---
    definitions_id = generate_unique_id("Definitions")
    collaboration_id = generate_unique_id("Collaboration")
    process_id = generate_unique_id("Process")
    diagram_id = generate_unique_id("BPMNDiagram")
    plane_id = generate_unique_id("BPMNPlane")

    semantic_participants = []
    semantic_lanes = []
    semantic_lane_sets = []
    semantic_flow_nodes = [] # Задачи, События, Шлюзы
    semantic_sequence_flows = []
    visual_shapes = []
    visual_edges = []
    element_map = {} # internal_id -> bpmn_element (семантический)
    element_coords = {} # bpmn_element_id -> {"x", "y", "width", "height"}
    agent_lane_map = {} # resolved_agent_word.lower() -> lane_id
    lane_flow_nodes = defaultdict(list) # lane_id -> list of flowNode IDs
    diverging_to_converging_map = {} # diverging_gw_internal_id -> converging_gw_bpmn_id


    # --- 2. Создание Участников (Пулов) и Дорожек (Lanes) ---
    print("Creating Pools and Lanes...")
    if not unique_agents:
        print(f"{Fore.YELLOW}Warning: No unique agents found. Creating a default pool and lane.{Fore.RESET}")
        unique_agents = ["Default Process"]

    main_pool_id = generate_unique_id("Participant")
    main_pool_name = unique_agents[0] if len(unique_agents) == 1 else "Main Process Collaboration"
    semantic_participants.append({"id": main_pool_id, "name": main_pool_name, "processRef": process_id})

    lane_set_id = generate_unique_id("LaneSet")
    for agent_name in unique_agents:
        lane_id = generate_unique_id("Lane")
        agent_key = agent_name.lower() if isinstance(agent_name, str) else str(agent_name)
        agent_lane_map[agent_key] = lane_id
        semantic_lanes.append({"id": lane_id, "name": agent_name, "flowNodeRef": []})

    if semantic_lanes:
        semantic_lane_sets.append({"id": lane_set_id, "lanes": [lane["id"] for lane in semantic_lanes]})
        process_has_laneset = True
    else:
        process_has_laneset = False
        default_lane_id = None
        print(f"{Fore.YELLOW}Warning: No lanes created, elements will not be assigned to lanes.{Fore.RESET}")


    # --- 3. Построение Графа Потока Управления (Семантика) ---
    print("Building semantic flow graph...")

    # --- 3.1 Объединение и сортировка ---
    all_elements_data = []
    for i, elem_info in enumerate(agent_task_pairs_or_loops):
        element_type = "unknown"; internal_id = None; start_index = None; data = elem_info
        if isinstance(elem_info, dict):
            if "task" in elem_info:
                element_type = "task"
                internal_id = elem_info["task"].get("task_id")
                if not internal_id: internal_id = f"AutoGenTask_{i}"
                start_index = elem_info["task"].get("start")
            elif "go_to" in elem_info:
                element_type = "loop"
                internal_id = f"LoopMarker_{i}"
                start_index = elem_info.get("start")
            else: continue
            if start_index is not None:
                all_elements_data.append({
                    "type": element_type, "internal_id": internal_id,
                    "original_index": i, "start": start_index, "data": data
                })
            else: print(f"{Fore.YELLOW}Warning: Element type '{element_type}' lacks start index. Skipping.{Fore.RESET}")
        else: continue

    gateways_data = parallel_gateways + exclusive_gateways
    for gw in gateways_data:
        gw_type = "parallel_gateway" if gw.get("id", "").startswith("PG") else "exclusive_gateway"
        start_index = gw.get("start"); gw_internal_id = gw.get("id")
        if start_index is not None and gw_internal_id:
            all_elements_data.append({
                "type": gw_type, "internal_id": gw_internal_id,
                "original_index": -1, "start": start_index, "data": gw
            })
        else: print(f"{Fore.YELLOW}Warning: Gateway {gw_internal_id or 'Unknown'} lacks start index or ID, skipping.{Fore.RESET}")

    all_elements_data.sort(key=lambda x: x.get("start", float('inf')))
    element_indices_map = {elem_data["internal_id"]: i for i, elem_data in enumerate(all_elements_data)}

    # --- 3.2 Создание BPMN Элементов (Узлы) ---
    print("Creating nodes...")

    def get_lane_for_element(element_data: dict, fallback_lane_id: str | None) -> str | None:
        agent_name = None; lane_id = None
        elem_type = element_data.get("type")
        internal_id_ctx = element_data.get("internal_id")

        agent_info = None
        if elem_type == "task": agent_info = element_data["data"].get("agent", {})
        elif elem_type == "loop": agent_info = element_data["data"].get("original_loop_agent", {})
        elif elem_type == "converging_gateway":
            diverging_gw_internal_id = element_data.get("diverging_gw_internal_id")
            if diverging_gw_internal_id:
                 diverging_gw_bpmn = element_map.get(diverging_gw_internal_id)
                 if diverging_gw_bpmn: return diverging_gw_bpmn.get("lane")

        if agent_info:
            agent_name = agent_info.get("resolved_word")
            if agent_name and isinstance(agent_name, str): lane_id = agent_lane_map.get(agent_name.lower())

        elif elem_type in ["exclusive_gateway", "parallel_gateway"]:
             current_index = element_indices_map.get(internal_id_ctx)
             if current_index is not None and current_index > 0:
                 for k in range(current_index - 1, -1, -1):
                     prev_elem_data = all_elements_data[k]
                     if prev_elem_data.get("type") != "loop":
                         prev_bpmn_elem = element_map.get(prev_elem_data.get("internal_id"))
                         if prev_bpmn_elem: lane_id = prev_bpmn_elem.get("lane"); break

        elif internal_id_ctx == "START" and semantic_lanes: return semantic_lanes[0]["id"]
        elif internal_id_ctx == "END" and semantic_lanes:
             last_flow_elem_data = next((e for e in reversed(all_elements_data) if e['type'] != 'loop'), None)
             if last_flow_elem_data:
                  last_bpmn_elem = element_map.get(last_flow_elem_data.get("internal_id"))
                  if last_bpmn_elem and "lane" in last_bpmn_elem: return last_bpmn_elem["lane"]
             return semantic_lanes[-1]["id"] # Fallback

        if not lane_id and fallback_lane_id: lane_id = fallback_lane_id
        return lane_id

    default_lane_id = semantic_lanes[0]["id"] if semantic_lanes else None

    # StartEvent
    start_event_id = generate_unique_id("StartEvent")
    start_event_data_ctx = {"type": "event", "internal_id": "START"}
    start_event = {"id": start_event_id, "type": "bpmn:StartEvent", "name": "Start"}
    start_lane = get_lane_for_element(start_event_data_ctx, default_lane_id)
    if start_lane: start_event["lane"] = start_lane; lane_flow_nodes[start_lane].append(start_event_id)
    semantic_flow_nodes.append(start_event); element_map["START"] = start_event

    # Основные узлы (Первый проход - создание)
    temp_element_refs = {}
    for index, elem_data in enumerate(all_elements_data):
        elem_type = elem_data["type"]; internal_id = elem_data["internal_id"]; data = elem_data["data"]; bpmn_element = None; bpmn_element_id = None
        if elem_type == "loop": continue

        if elem_type == "task":
            bpmn_element_id = generate_unique_id("UserTask"); task_name = data.get("task", {}).get("word", "Unnamed Task"); bpmn_element = {"id": bpmn_element_id, "type": "bpmn:UserTask", "name": task_name}; original_task_id = internal_id
            if original_task_id: element_map[original_task_id] = bpmn_element
        elif elem_type == "exclusive_gateway":
            bpmn_element_id = internal_id  # Используем ID шлюза из данных (EG0, EG1...)
            gw_name = data.get("name", f"Choice {bpmn_element_id}")
            # Убираем префикс "Assistant:" если он есть
            if gw_name.startswith("Assistant:"):
                gw_name = gw_name[len("Assistant:"):].strip()
            bpmn_element = {"id": bpmn_element_id, "type": "bpmn:ExclusiveGateway", "name": gw_name}
            element_map[internal_id] = bpmn_element  # Сохраняем связь
        elif elem_type == "parallel_gateway":
            bpmn_element_id = internal_id  # Используем ID шлюза из данных (PG0, PG1...)
            gw_name = data.get("name", f"Parallel {bpmn_element_id}")
            # Убираем префикс "Assistant:" если он есть
            if gw_name.startswith("Assistant:"):
                gw_name = gw_name[len("Assistant:"):].strip()
            bpmn_element = {"id": bpmn_element_id, "type": "bpmn:ParallelGateway", "name": gw_name}
            element_map[internal_id] = bpmn_element  # Сохраняем связь
        if bpmn_element:
             semantic_flow_nodes.append(bpmn_element)
             if bpmn_element_id: temp_element_refs[bpmn_element_id] = internal_id

    # Второй проход для присвоения дорожек
    for bpmn_node in semantic_flow_nodes:
        node_id = bpmn_node.get("id")
        if bpmn_node['type'] == "bpmn:StartEvent" or "lane" in bpmn_node: continue
        internal_id_ref = temp_element_refs.get(node_id)
        if not internal_id_ref: continue
        elem_data_ref = find_element_by_internal_id(all_elements_data, internal_id_ref)
        if elem_data_ref:
            current_lane_id = get_lane_for_element(elem_data_ref, default_lane_id)
            if current_lane_id: bpmn_node["lane"] = current_lane_id; lane_flow_nodes[current_lane_id].append(node_id)

    # EndEvent
    end_event_id = generate_unique_id("EndEvent")
    end_event_data_ctx = {"type": "event", "internal_id": "END"}
    end_event = {"id": end_event_id, "type": "bpmn:EndEvent", "name": "End"}
    end_lane = get_lane_for_element(end_event_data_ctx, default_lane_id)
    if end_lane: end_event["lane"] = end_lane; lane_flow_nodes[end_lane].append(end_event_id)
    semantic_flow_nodes.append(end_event); element_map["END"] = end_event


    # --- 3.3 Создание Потоков Управления (Новый Алгоритм v4 - Явный обход) ---
    print("Creating sequence flows (Revised Algorithmic Approach v4)...")
    created_flows = set()
    semantic_sequence_flows = []
    processed_sources = set() # BPMN ID источников, у которых уже создан исходящий поток

    def add_flow(source_id, target_id, name=None):
        if not source_id or not target_id or source_id == target_id: return False
        flow_tuple = (source_id, target_id)
        if flow_tuple in created_flows: return False
        flow_id = generate_unique_id("Flow"); flow_data = {"id": flow_id, "sourceRef": source_id, "targetRef": target_id}
        if name and isinstance(name, str) and name.strip(): flow_data["name"] = name.strip()
        semantic_sequence_flows.append(flow_data); created_flows.add(flow_tuple);
        processed_sources.add(source_id) # Помечаем источник как обработанный
        # print(f"DEBUG Flow: Added {source_id} -> {target_id} (Name: {name})")
        return True

    # --- Основной цикл обработки элементов ---
    last_processed_bpmn_id = start_event_id # Начинаем со StartEvent
    processed_element_indices = set() # Индексы обработанных элементов в all_elements_data

    for i, current_elem_data in enumerate(all_elements_data):
        current_internal_id = current_elem_data["internal_id"]
        current_bpmn_elem = element_map.get(current_internal_id)
        elem_type = current_elem_data["type"]

        if elem_type == "loop" or not current_bpmn_elem: continue
        current_bpmn_id = current_bpmn_elem["id"]

        # --- Обработка Расходящихся Шлюзов ---
        is_diverging_gateway = elem_type in ["exclusive_gateway", "parallel_gateway"]
        if is_diverging_gateway:
            print(f"  Processing Diverging Gateway: {current_bpmn_elem.get('name', current_bpmn_id)}")
            gateway_data = current_elem_data["data"]; conditions = gateway_data.get("conditions", [])

            # Получаем ветки от LLM
            subsequent_tasks_for_llm = []
            for k in range(i + 1, len(all_elements_data)):
                elem_k_data = all_elements_data[k]; elem_k_type = elem_k_data["type"]; elem_k_internal_id = elem_k_data["internal_id"]
                if elem_k_type in ["task", "exclusive_gateway", "parallel_gateway"]:
                     task_text = elem_k_data["data"].get("task", {}).get("word", elem_k_internal_id) if elem_k_type == "task" else elem_k_data["data"].get("name", elem_k_internal_id)
                     subsequent_tasks_for_llm.append({"id": elem_k_internal_id, "text": task_text})
                if len(subsequent_tasks_for_llm) >= 10: break
            context_text = original_text[max(0, gateway_data.get("start", 0) - 150) : gateway_data.get("end", len(original_text)) + 150]

            branch_starts = {} # cond_idx -> immediate_task_internal_id
            llm_call_success = False
            print(f"    DEBUG: Calling LLM for gateway {current_internal_id}...") # Отладка
            if prompts.model_loaded and subsequent_tasks_for_llm:
                try:
                    llm_response_str = prompts.determine_gateway_flow(current_internal_id, conditions, subsequent_tasks_for_llm, context_text)
                    if llm_response_str:
                         llm_flow_data = json.loads(llm_response_str)
                         print(f"    LLM suggested branch starts for {current_internal_id}: {llm_flow_data.get('branches')}")
                         for branch_info in llm_flow_data.get("branches", []):
                             cond_idx = branch_info.get("condition_index")
                             immediate_task_id = branch_info.get("immediate_task_id")
                             if immediate_task_id and element_map.get(immediate_task_id):
                                 if elem_type == "parallel_gateway": branch_starts[len(branch_starts)] = immediate_task_id
                                 elif cond_idx is not None: branch_starts[cond_idx] = immediate_task_id
                                 llm_call_success = True
                             else: print(f"{Fore.YELLOW}Warning: LLM suggested immediate task '{immediate_task_id}' for branch {cond_idx} not found or invalid.{Fore.RESET}")
                    else: print(f"{Fore.YELLOW}Warning: LLM returned empty response for {current_internal_id}.{Fore.RESET}")
                except Exception as e: print(f"{Fore.RED}Error parsing LLM flow response for {current_internal_id}: {e}{Fore.RESET}")
            else: print(f"{Fore.YELLOW}Warning: Skipping LLM call for {current_internal_id} (model loaded: {prompts.model_loaded}, tasks: {bool(subsequent_tasks_for_llm)}){Fore.RESET}")

            if not llm_call_success:
                 print(f"{Fore.YELLOW}Warning: Could not determine branch starts for {current_internal_id}. Using basic fallback.{Fore.RESET}")
                 next_elem_data = find_first_element_after(all_elements_data, i, ["task", "exclusive_gateway", "parallel_gateway"])
                 target_bpmn_id = element_map.get(next_elem_data["internal_id"], {}).get('id') if next_elem_data else end_event_id
                 if target_bpmn_id: add_flow(current_bpmn_id, target_bpmn_id)
                 last_processed_bpmn_id = target_bpmn_id # Обновляем последний обработанный
                 continue # Переходим к следующему элементу основного цикла

            # --- Создаем Сходящийся Шлюз ---
            converging_gateway_id = generate_unique_id("ConvergeGateway")
            converging_gateway_type = "bpmn:ExclusiveGateway" if elem_type == "exclusive_gateway" else "bpmn:ParallelGateway"
            converging_gateway = {"id": converging_gateway_id, "type": converging_gateway_type, "name": ""}
            semantic_flow_nodes.append(converging_gateway); element_map[converging_gateway_id] = converging_gateway
            diverging_to_converging_map[current_internal_id] = converging_gateway_id
            print(f"    Created Converging Gateway: {converging_gateway_id} for {current_internal_id}")

            # --- Обработка Каждой Ветки ---
            branch_end_elements = [] # Сохраняем последние элементы каждой ветки
            max_branch_elem_index = i # Отслеживаем максимальный индекс элемента в ветках

            for cond_idx, immediate_task_internal_id in branch_starts.items():
                first_task_bpmn = element_map.get(immediate_task_internal_id)
                if not first_task_bpmn: continue

                condition_name = conditions[cond_idx] if elem_type == "exclusive_gateway" and cond_idx < len(conditions) else None

                # 1. Поток: Расходящийся -> Первая задача ветки
                add_flow(current_bpmn_id, first_task_bpmn["id"], condition_name)
                first_task_index = element_indices_map.get(immediate_task_internal_id)
                if first_task_index is not None:
                    processed_element_indices.add(first_task_index)
                    max_branch_elem_index = max(max_branch_elem_index, first_task_index)

                # 2. Итеративный обход ветки (простой вариант: пока только первый элемент)
                last_valid_element_in_branch_bpmn = first_task_bpmn
                # TODO: Реализовать более сложный обход ветки при необходимости

                branch_end_elements.append(last_valid_element_in_branch_bpmn["id"])

            # 3. Соединяем концы веток со сходящимся шлюзом
            for branch_end_id in branch_end_elements:
                 add_flow(branch_end_id, converging_gateway_id)

            # 4. Соединяем Сходящийся Шлюз со следующим элементом
            next_elem_data = find_first_element_after(all_elements_data, max_branch_elem_index, ["task", "exclusive_gateway", "parallel_gateway"])
            target_after_converging_id = element_map.get(next_elem_data["internal_id"], {}).get('id') if next_elem_data else end_event_id

            if target_after_converging_id:
                add_flow(converging_gateway_id, target_after_converging_id)
                last_processed_bpmn_id = target_after_converging_id
                # Помечаем следующий элемент как обработанный
                if next_elem_data:
                     next_index = element_indices_map.get(next_elem_data["internal_id"])
                     if next_index is not None: processed_element_indices.add(next_index)

            # Присваиваем дорожку сходящемуся шлюзу
            conv_lane = get_lane_for_element({"type": "converging_gateway", "diverging_gw_internal_id": current_internal_id}, default_lane_id)
            if conv_lane: converging_gateway["lane"] = conv_lane; lane_flow_nodes[conv_lane].append(converging_gateway_id)

        # --- Обработка Обычных Задач (если их вход еще не создан) ---
        elif i not in processed_element_indices:
             # Ищем следующий элемент
             next_elem_data = find_first_element_after(all_elements_data, i, ["task", "exclusive_gateway", "parallel_gateway"])
             target_bpmn_id = element_map.get(next_elem_data["internal_id"], {}).get('id') if next_elem_data else end_event_id

             # Соединяем текущий элемент со следующим (или концом)
             if target_bpmn_id:
                 add_flow(current_bpmn_id, target_bpmn_id)
                 last_processed_bpmn_id = target_bpmn_id
                 # Помечаем следующий элемент как обработанный
                 if next_elem_data:
                      next_index = element_indices_map.get(next_elem_data["internal_id"])
                      if next_index is not None: processed_element_indices.add(next_index)


    # --- 3.5 Обработка Циклов ---
    print("Processing loops...")
    for elem_data in all_elements_data:
        if elem_data["type"] == "loop":
            data = elem_data["data"]; target_task_internal_id = data.get("go_to"); source_bpmn_elem = None
            marker_index = element_indices_map.get(elem_data["internal_id"])
            if marker_index is None: continue
            for k in range(marker_index - 1, -1, -1):
                prev_elem_data = all_elements_data[k]
                if prev_elem_data["type"] != "loop": source_bpmn_elem = element_map.get(prev_elem_data.get("internal_id")); break
            target_bpmn_task = element_map.get(target_task_internal_id)
            if source_bpmn_elem and target_bpmn_task: add_flow(source_bpmn_elem["id"], target_bpmn_task["id"], "Loop back")
            else: print(f"{Fore.YELLOW}Warning: Could not create loop flow. Source: {source_bpmn_elem.get('id') if source_bpmn_elem else 'NotFound'}, Target: {target_bpmn_task.get('id') if target_bpmn_task else 'NotFound'}{Fore.RESET}")

    # --- 3.6 Соединение "висячих" узлов с EndEvent ---
    print("Connecting remaining dangling ends to End Event...")
    # flow_sources = {flow['sourceRef'] for flow in semantic_sequence_flows} # Используем processed_sources
    for node in semantic_flow_nodes:
        node_id = node['id']; node_type = node['type']
        if node_type in ["bpmn:StartEvent", "bpmn:EndEvent"]: continue
        if node_id not in processed_sources: # Если у узла не было создано исходящего потока
            # Исключаем сходящиеся шлюзы
            is_converging_gateway = node_id in diverging_to_converging_map.values()
            if not is_converging_gateway:
                 print(f"  Connecting dangling node {node_id} ({node.get('name','')}, {node_type}) to End Event.")
                 add_flow(node_id, end_event_id)

    # --- Финальная чистка и обновление ---
    for lane in semantic_lanes:
        lane["flowNodeRef"] = lane_flow_nodes.get(lane["id"], [])


    # --- 4. Генерация Визуальной Информации (BPMNDI) ---
    # (Код раскладки оставлен без изменений)
    print("Generating visual layout...")
    node_levels = defaultdict(int); level_nodes = defaultdict(list); max_level = 0;
    queue = [(start_event_id, 0)] if start_event_id else []; visited_for_layout = {start_event_id} if start_event_id else set();
    if start_event_id: node_levels[start_event_id] = 0; level_nodes[0].append(start_event_id);
    processed_edges_layout = set()

    while queue:
        current_id_layout, level_layout = queue.pop(0); max_level = max(max_level, level_layout);
        out_flows = [f for f in semantic_sequence_flows if f['sourceRef'] == current_id_layout]
        out_flows.sort(key=lambda f: f['targetRef'])

        for flow_layout in out_flows:
            target_id_layout = flow_layout['targetRef']; flow_tuple_layout = (flow_layout['sourceRef'], target_id_layout);
            target_level_current = node_levels.get(target_id_layout, -1)
            is_loop_edge = target_level_current != -1 and target_level_current <= level_layout
            if is_loop_edge or flow_tuple_layout in processed_edges_layout: continue

            new_level_layout = level_layout + 1
            if target_id_layout not in visited_for_layout or new_level_layout > target_level_current:
                if target_id_layout in visited_for_layout and target_level_current != -1:
                    if target_id_layout in level_nodes.get(target_level_current,[]): level_nodes[target_level_current].remove(target_id_layout)
                node_levels[target_id_layout] = new_level_layout
                if target_id_layout not in level_nodes.get(new_level_layout, []): level_nodes[new_level_layout].append(target_id_layout)
                if target_id_layout not in visited_for_layout:
                     visited_for_layout.add(target_id_layout); queue.append((target_id_layout, new_level_layout))
            processed_edges_layout.add(flow_tuple_layout)

    unvisited_nodes = [n['id'] for n in semantic_flow_nodes if n['id'] not in node_levels]
    nodes_processed_in_fallback = True
    while unvisited_nodes and nodes_processed_in_fallback:
        nodes_processed_in_fallback = False; remaining_nodes = []
        for node_id in unvisited_nodes:
            in_flows = [f for f in semantic_sequence_flows if f['targetRef'] == node_id]
            pred_levels = [node_levels.get(f['sourceRef'], -1) for f in in_flows]
            valid_pred_levels = [lvl for lvl in pred_levels if lvl != -1]
            if valid_pred_levels:
                node_level = max(valid_pred_levels) + 1; node_levels[node_id] = node_level;
                if node_id not in level_nodes.get(node_level, []): level_nodes[node_level].append(node_id)
                max_level = max(max_level, node_level); nodes_processed_in_fallback = True;
            else:
                if node_id == end_event_id and not in_flows:
                    node_level = 1; node_levels[node_id] = node_level;
                    if node_id not in level_nodes.get(node_level, []): level_nodes[node_level].append(node_id)
                    max_level = max(max_level, node_level); nodes_processed_in_fallback = True;
                elif node_id != start_event_id: remaining_nodes.append(node_id)
        unvisited_nodes = remaining_nodes
        if not nodes_processed_in_fallback and unvisited_nodes:
             print(f"{Fore.YELLOW}Warning: Could not determine level for nodes: {unvisited_nodes}. Layout might be incorrect.{Fore.RESET}")
             fallback_level = max_level + 1
             for node_id in unvisited_nodes:
                 node_levels[node_id] = fallback_level
                 if node_id not in level_nodes.get(fallback_level, []): level_nodes[fallback_level].append(node_id)
             max_level = fallback_level; break

    element_dims = {}; element_coords = {};
    for element in semantic_flow_nodes:
        elem_id = element["id"]; elem_type = element["type"]; width = DEFAULT_WIDTH; height = DEFAULT_HEIGHT;
        if "Event" in elem_type: width = EVENT_WIDTH; height = EVENT_HEIGHT;
        elif "Gateway" in elem_type: width = GATEWAY_WIDTH; height = GATEWAY_HEIGHT;
        element_dims[elem_id] = (width, height)

    level_start_x = {}; current_x = POOL_PADDING_X + LANE_HEADER_WIDTH + HORIZONTAL_SPACING // 2
    for level in range(max_level + 1):
        level_start_x[level] = current_x; max_w_on_level = 0;
        nodes_on_this_level = level_nodes.get(level, [])
        valid_nodes_on_level = [nid for nid in nodes_on_this_level if nid in element_dims]
        if valid_nodes_on_level: max_w_on_level = max(element_dims[nid][0] for nid in valid_nodes_on_level);
        current_x += max(max_w_on_level, GATEWAY_WIDTH) + HORIZONTAL_SPACING
    max_diagram_width = current_x + POOL_PADDING_X

    lane_y_current = {}; lane_heights = {}; initial_lane_y_starts = {};
    current_y_offset = POOL_PADDING_Y
    for lane in semantic_lanes:
        lane_id = lane['id']; initial_lane_y_starts[lane_id] = current_y_offset; lane_y_current[lane_id] = current_y_offset + LANE_PADDING_Y; lane_heights[lane_id] = LANE_MIN_HEIGHT; current_y_offset += lane_heights[lane_id];

    processed_layout_nodes = set()
    all_levels = sorted(level_nodes.keys())
    for level in all_levels:
        nodes_in_level = level_nodes.get(level, []); nodes_by_lane = defaultdict(list);
        for node_id in nodes_in_level:
             node_info = find_element_by_bpmn_id(semantic_flow_nodes, node_id)
             if node_info:
                  lane_id = node_info.get('lane')
                  if lane_id: nodes_by_lane[lane_id].append(node_id)

        for lane_id, node_ids_in_lane in nodes_by_lane.items():
            node_ids_in_lane.sort()
            base_y_for_level = lane_y_current.get(lane_id, initial_lane_y_starts.get(lane_id, POOL_PADDING_Y) + LANE_PADDING_Y)
            current_level_max_y = base_y_for_level

            for node_id in node_ids_in_lane:
                  if node_id in processed_layout_nodes: continue
                  if node_id not in element_dims: continue

                  node_level = node_levels.get(node_id, level)
                  x = level_start_x.get(node_level, POOL_PADDING_X)
                  width, height = element_dims[node_id]
                  y = base_y_for_level

                  center_x = x
                  final_x = center_x - width // 2

                  element_coords[node_id] = {"x": final_x, "y": y, "width": width, "height": height}
                  processed_layout_nodes.add(node_id)
                  base_y_for_level += height + VERTICAL_SPACING
                  current_level_max_y = max(current_level_max_y, y + height)

            if node_ids_in_lane:
                 lane_y_current[lane_id] = base_y_for_level
                 required_height = (current_level_max_y - initial_lane_y_starts.get(lane_id, 0)) + LANE_PADDING_Y
                 lane_heights[lane_id] = max(lane_heights.get(lane_id, 0), required_height, LANE_MIN_HEIGHT)

    lane_y_positions = {}; current_y = POOL_PADDING_Y; max_pool_height = POOL_PADDING_Y;
    for lane in semantic_lanes:
        lane_id = lane['id']; final_lane_height = lane_heights.get(lane_id, LANE_MIN_HEIGHT); lane_y_positions[lane_id] = current_y; current_y += final_lane_height;
    max_pool_height = current_y

    visual_shapes = []
    pool_width = max_diagram_width - POOL_PADDING_X
    pool_height = max_pool_height - POOL_PADDING_Y
    visual_shapes.append({ "id": generate_unique_id("Shape_Participant"), "bpmnElement": main_pool_id, "isHorizontal": True, "isExpanded": True, "bounds": {"x": POOL_PADDING_X, "y": POOL_PADDING_Y, "width": pool_width, "height": pool_height} })
    lane_width = pool_width - LANE_HEADER_WIDTH
    for lane in semantic_lanes:
        lane_id = lane['id']; lane_height = lane_heights.get(lane_id, LANE_MIN_HEIGHT); lane_y = lane_y_positions.get(lane_id, POOL_PADDING_Y);
        visual_shapes.append({ "id": generate_unique_id("Shape_Lane"), "bpmnElement": lane_id, "isHorizontal": True, "bounds": {"x": POOL_PADDING_X + LANE_HEADER_WIDTH, "y": lane_y, "width": lane_width, "height": lane_height} })
    for node in semantic_flow_nodes:
         node_id = node['id'];
         if node_id in element_coords: visual_shapes.append({"id": generate_unique_id(f"Shape_{node_id}"), "bpmnElement": node_id, "bounds": element_coords[node_id]})
         else: print(f"{Fore.YELLOW}Warning: Node {node_id} ('{node.get('name','')}') lacks coordinates. Shape will not be generated.{Fore.RESET}")

    print("Creating visual edges...")
    visual_edges = []
    for flow in semantic_sequence_flows:
        flow_id = flow["id"]; source_id = flow["sourceRef"]; target_id = flow["targetRef"]; source_coords = element_coords.get(source_id); target_coords = element_coords.get(target_id); waypoints = []
        if source_coords and target_coords:
            cx_source, cy_source = get_element_center(source_coords).values(); cx_target, cy_target = get_element_center(target_coords).values(); source_level = node_levels.get(source_id, -1); target_level = node_levels.get(target_id, -1);
            start_point = {"x": source_coords["x"] + source_coords["width"], "y": cy_source}; end_point = {"x": target_coords["x"], "y": cy_target};
            is_loop_back = target_level < source_level and target_level != -1; is_vertical_in_level = target_level == source_level and source_level != -1 and source_id != target_id;

            if is_loop_back:
                 start_point = {"x": cx_source, "y": source_coords["y"] + source_coords["height"]}; end_point = {"x": cx_target, "y": target_coords["y"]};
                 waypoints.append(start_point); mid_y = max(start_point['y'], end_point['y'] + target_coords["height"]) + VERTICAL_SPACING // 2; h_offset = GATEWAY_WIDTH + 20;
                 waypoints.append({"x": start_point["x"], "y": mid_y}); waypoints.append({"x": end_point["x"] - h_offset, "y": mid_y}); waypoints.append({"x": end_point["x"] - h_offset, "y": end_point["y"]}); waypoints.append(end_point)
            elif is_vertical_in_level:
                 if cy_target > cy_source: start_point = {"x": cx_source, "y": source_coords["y"] + source_coords["height"]}; end_point = {"x": cx_target, "y": target_coords["y"]};
                 else: start_point = {"x": cx_source, "y": source_coords["y"]}; end_point = {"x": cx_target, "y": target_coords["y"] + target_coords["height"]};
                 waypoints.append(start_point); mid_x_offset = (DEFAULT_WIDTH + HORIZONTAL_SPACING) // 4; mid_x = max(source_coords["x"]+source_coords["width"], target_coords["x"]+target_coords["width"]) + mid_x_offset;
                 waypoints.append({"x": mid_x, "y": start_point["y"]}); waypoints.append({"x": mid_x, "y": end_point["y"]}); waypoints.append(end_point)
            else:
                 waypoints.append(start_point);
                 if abs(start_point["y"] - end_point["y"]) > 5:
                     mid_x = (start_point["x"] + end_point["x"]) // 2; mid_x = max(start_point["x"] + 10, mid_x); mid_x = min(end_point["x"] - 10, mid_x);
                     waypoints.append({"x": mid_x, "y": start_point["y"]}); waypoints.append({"x": mid_x, "y": end_point["y"]});
                 waypoints.append(end_point);

            edge_data = {"id": generate_unique_id(f"Edge_{flow_id}"), "bpmnElement": flow_id, "waypoints": waypoints}
            flow_name = flow.get("name")
            if flow_name:
                 label_x = waypoints[len(waypoints)//2]["x"] + 10; label_y = waypoints[len(waypoints)//2]["y"] - 10; label_width = max(50, len(flow_name) * 7)
                 edge_data["label"] = { "bounds": { "x": label_x, "y": label_y, "width": label_width, "height": 14 } }
            visual_edges.append(edge_data)
        else: print(f"{Fore.YELLOW}Warning: Missing coords for flow {flow_id} (Source: {source_id} {'OK' if source_coords else 'MISSING'}, Target: {target_id} {'OK' if target_coords else 'MISSING'}). Skipping edge generation.{Fore.RESET}")

    # --- 5. Сборка финального JSON ---
    print("Assembling final JSON...")
    final_json = {
        "definitions": {
            "id": definitions_id, "targetNamespace": "http://bpmn.io/schema/bpmn", "exporter": "Python BPMN Generator", "exporterVersion": "0.13", # Версия
            "collaboration": {"id": collaboration_id, "participants": semantic_participants},
            "process": {
                "id": process_id, "isExecutable": False, "name": process_id, **({"laneSets": semantic_lane_sets} if process_has_laneset else {}),
                "flowElements": semantic_flow_nodes, "sequenceFlows": semantic_sequence_flows
            }
        },
        "diagrams": [{"id": diagram_id, "plane": {"id": plane_id, "bpmnElement": collaboration_id, "planeElement": visual_shapes + visual_edges}}]
    }

    print(f"{Fore.GREEN}BPMN JSON structure created successfully.{Fore.RESET}")
    return final_json

# --- END OF FILE create_bpmn_structure.py ---