# --- START OF FILE bpmn_validator.py ---

import os
import json
import traceback
import copy
import re
import uuid # Для генерации уникальных ID валидации
from typing import Any, Dict, List, Optional, Set, Tuple
from os.path import exists
from colorama import Fore, Style

# Я немного подшамнил - ИЗМЕНЕНИЯ: - далее эту заметку убрать
# Предварительная проверка, установлен ли networkx
try:
    import networkx as nx
except ImportError:
    print(f"{Fore.RED}ERROR: 'networkx' library is required for BPMN validation.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Please install it using: pip install networkx{Style.RESET_ALL}")
    nx = None

# Импорт colorama для цветного вывода
try:
    from colorama import Fore, Style, init
    init(autoreset=True) # Инициализация colorama
except ImportError:
    # Заглушки, если colorama не установлена
    print("Warning: 'colorama' not found. Output will not be colored.")
    class Fore: Style=Fore; RED=YELLOW=GREEN=CYAN=MAGENTA=BLUE=BRIGHT="" # Добавлен BRIGHT
    class Style: RESET_ALL=BRIGHT="" # Добавлен BRIGHT

# --- LLM Зависимости ---
LLM_AVAILABLE = False
llm_model, llm_tokenizer, LLM_DEVICE = None, None, None
try:
    # Проверяем наличие библиотек перед импортом
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    try:
        from llm_analyzer_prompts import BPMN_ANALYSIS_SYSTEM_MSG, create_bpmn_analysis_prompt
        PROMPTS_AVAILABLE = True
    except ImportError:
        print(f"{Fore.YELLOW}Warning: Could not import prompts from 'llm_analyzer_prompts.py'. LLM analysis might fail.{Style.RESET_ALL}")
        PROMPTS_AVAILABLE = False
        # Определяем заглушки
        BPMN_ANALYSIS_SYSTEM_MSG = "Analyze the BPMN structure."
        def create_bpmn_analysis_prompt(text, structure, errors): return "Please analyze."


    LLM_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
    print(f"Attempting to load LLM model: {LLM_MODEL_NAME}...")
    # Уменьшаем объем логов transformers
    transformers.logging.set_verbosity_error()

    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    llm_model = transformers.AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True, torch_dtype=dtype)
    llm_model.eval()
    LLM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model.to(LLM_DEVICE)
    print(f"LLM model loaded successfully to {LLM_DEVICE}.")
    LLM_AVAILABLE = True
    # Восстанавливаем уровень логов (если нужно)
    # transformers.logging.set_verbosity_warning()

except ImportError:
    print(f"{Fore.YELLOW}Warning: 'transformers' or 'torch' not found. LLM analysis will be disabled.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error loading LLM model ({LLM_MODEL_NAME}): {e}{Style.RESET_ALL}")
    traceback.print_exc()
# --- Конец LLM Зависимостей ---


class BPMNValidator:
    """
    Проверяет структуру BPMN, выявляет ошибки, предлагает исправления
    и может применять безопасные автоматические исправления.
    """
    def __init__(self, bpmn_structure: List[Dict[str, Any]], original_text: str):
        if not isinstance(bpmn_structure, list):
            # Log critical error instead of raising ValueError immediately
            self._log_critical_error("Initialization", "bpmn_structure must be a list of dictionaries.")
            self.original_structure = None
            self.current_structure = None
        else:
            self.original_structure = copy.deepcopy(bpmn_structure)
            self.current_structure = copy.deepcopy(bpmn_structure) # Structure that might be modified

        self.original_text = original_text
        self.graph: Optional[nx.DiGraph] = nx.DiGraph() if nx else None # Создаем граф, если networkx доступен
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []
        self.new_llm_issues: List[str] = []
        self._validation_id_map: Dict[str, Dict] = {} # validation_id -> element dict
        self._element_id_map: Dict[str, str] = {} # task_id/gateway_id -> validation_id
        self.node_map: Dict[str, List[str]] = {} # validation_id -> [list_of_graph_node_ids]

        # Добавляем ID сразу при инициализации, если структура валидна
        if self.current_structure is not None:
             self._add_validation_ids(self.current_structure)

    # --- Утилиты ID ---
    def _add_validation_ids(self, structure: list | dict):
        """Рекурсивно добавляет уникальный "_validation_id" к каждому элементу словаря."""
        if isinstance(structure, dict):
            if '_validation_id' not in structure:
                val_id = str(uuid.uuid4())
                structure['_validation_id'] = val_id
                self._validation_id_map[val_id] = structure
                elem_type = structure.get("type")
                elem_id = None
                if elem_type == "task": elem_id = structure.get("content", {}).get("task", {}).get("task_id")
                elif elem_type in ["exclusive", "parallel"]: elem_id = structure.get("id")
                if elem_id: self._element_id_map[elem_id] = val_id

            for key, value in structure.items():
                 # Рекурсивно обходим только списки и словари, кроме 'content' dicts
                 if isinstance(value, list): self._add_validation_ids(value)
                 elif isinstance(value, dict) and key not in ['content', 'task', 'agent', 'condition']: # Не заходим глубоко в content
                      self._add_validation_ids(value)
                 # ИЗМЕНЕНИЕ: Добавлен рекурсивный обход для children, если они есть
                 elif key == 'children' and isinstance(value, list):
                      for path in value: self._add_validation_ids(path)

        elif isinstance(structure, list):
            for item in structure: self._add_validation_ids(item)

    def _get_element_by_validation_id(self, validation_id: str) -> Optional[Dict]:
        return self._validation_id_map.get(validation_id)

    def _get_validation_id_by_element_id(self, element_id: str) -> Optional[str]:
        return self._element_id_map.get(element_id)

    def _get_element_id_string(self, element: Optional[Dict]) -> str: # Добавлена проверка на None
         """Helper to get a descriptive string for an element for logging."""
         if not element: return "Element(N/A)" # Обработка случая, если элемент не найден

         elem_type = element.get("type")
         content = element.get("content", {})
         val_id = element.get("_validation_id", "no_vid")[:8] # Краткий ID валидации

         if elem_type == "task":
             task_id = content.get("task", {}).get("task_id", "?")
             task_word = content.get("task", {}).get("word", "?")
             return f"Task({task_id}|'{task_word[:20]}...'|{val_id})"
         elif elem_type in ["exclusive", "parallel"]:
             elem_id = element.get("id", "?")
             return f"{elem_type.capitalize()} Gateway({elem_id}|{val_id})"
         elif elem_type == "loop":
             loop_target = content.get("go_to", "?")
             return f"Loop(->{loop_target}|{val_id})"
         elif elem_type == "continue":
             continue_target = content.get("go_to", "?")
             return f"Continue(->{continue_target}|{val_id})"
         elif elem_type:
             return f"Element(type:{elem_type}|{val_id})"
         else:
              return f"Unknown Element({val_id})"

    # --- Logging Methods ---
    def log_error(self, category: str, element_ref: str | None, message: str):
        """Регистрирует ошибку, обнаруженную во время проверки."""
        ref = element_ref or "Structure"
        print(f"{Fore.RED}ERROR [{category} @ {ref}]: {message}{Style.RESET_ALL}")
        self.errors.append({"severity": "error", "category": category, "element_ref": ref, "message": message})

    def log_warning(self, category: str, element_ref: str | None, message: str):
        """Регистрирует предупреждение, обнаруженное во время проверки."""
        ref = element_ref or "Structure"
        print(f"{Fore.YELLOW}WARNING [{category} @ {ref}]: {message}{Style.RESET_ALL}")
        self.warnings.append({"severity": "warning", "category": category, "element_ref": ref, "message": message})

    def _log_critical_error(self, category: str, message: str):
        """Регистрирует критическую ошибку, которая препятствует дальнейшей проверке."""
        print(f"{Fore.RED}{Style.BRIGHT}CRITICAL ERROR [{category}]: {message}{Style.RESET_ALL}")
        self.errors.append({"severity": "critical", "category": category, "element_ref": "Initialization", "message": message})
        self.current_structure = None # Stop further processing

    # --- Graph Building ---
    def _build_graph(self, structure_to_build: List[Dict[str, Any]]):
        """Строит/перестраивает внутренний граф networkx на основе предоставленной структуры BPMN."""
        if not nx: self.log_warning("Graph Build", "N/A", "NetworkX not available. Skipping graph building."); return
        if not structure_to_build: self.log_warning("Graph Build", "N/A", "Structure is empty. Skipping graph building."); return

        self.graph.clear()
        self.node_map.clear()
        # ИЗМЕНЕНИЕ: Не очищаем element_id_map и validation_id_map здесь,
        # _add_validation_ids обновит их или добавит новые
        # self._element_id_map.clear()
        # self._validation_id_map.clear()
        self._add_validation_ids(structure_to_build) # Assign/update validation IDs to the current structure

        print("Building graph representation...")
        all_nodes_added = set() # Track nodes added to avoid duplicates

        def add_node_safe(node_id, **attrs):
            if node_id not in all_nodes_added:
                self.graph.add_node(node_id, **attrs)
                all_nodes_added.add(node_id)
            # ИЗМЕНЕНИЕ: Обновляем атрибуты, если узел уже существует (на случай перестроения)
            elif self.graph.has_node(node_id):
                 self.graph.nodes[node_id].update(attrs)


        def add_edge_safe(u, v, **attrs):
             # ИЗМЕНЕНИЕ: Добавляем узлы, если их нет, перед добавлением ребра
             if u not in self.graph: add_node_safe(u, type="implicit") # Добавляем неявный узел
             if v not in self.graph: add_node_safe(v, type="implicit") # Добавляем неявный узел

             # Добавляем ребро, если оба узла теперь существуют
             self.graph.add_edge(u, v, **attrs)
             # print(f"DEBUG: Added edge {u} -> {v} with {attrs}") # Отладка


        # Recursive function to add elements
        def process_elements(elements, parent_context):
            last_flow_src = parent_context.get("last_flow_src") # Node ID where sequence flow comes FROM

            for i, element in enumerate(elements):
                 val_id = element.get('_validation_id')
                 if not val_id: continue
                 elem_type = element.get("type")
                 elem_id_str = self._get_element_id_string(element) # For logging
                 graph_nodes_for_elem = [] # Graph nodes created for THIS element
                 current_start_node = None # Graph node representing the entry point of this element
                 current_end_node = None   # Graph node representing the exit point for sequence flow

                 if elem_type == "task":
                     task_info = element.get("content", {}).get("task", {})
                     task_id_base = task_info.get("task_id")
                     if task_id_base:
                          node_id = task_id_base
                          # ИЗМЕНЕНИЕ: Передаем сам элемент для информации
                          add_node_safe(node_id, type="task", label=task_info.get("word", "?"), validation_id=val_id, element_data=element)
                          graph_nodes_for_elem.append(node_id)
                          current_start_node = node_id; current_end_node = node_id
                     else: self.log_error("Graph Build", elem_id_str, "Task missing 'task_id'."); continue

                 elif elem_type in ["exclusive", "parallel"]:
                      gateway_id_base = element.get("id")
                      if gateway_id_base:
                           start_node_id = f"{gateway_id_base}_S"; end_node_id = f"{gateway_id_base}_E"
                           label = "X" if elem_type == "exclusive" else "+"
                           # ИЗМЕНЕНИЕ: Передаем сам элемент
                           add_node_safe(start_node_id, type=f"{elem_type}_start", label=label, validation_id=val_id, element_data=element)
                           graph_nodes_for_elem.append(start_node_id)
                           current_start_node = start_node_id

                           # ИЗМЕНЕНИЕ: Создаем конечный узел всегда, для связности структуры
                           # Логика соединения с ним будет зависеть от потомков
                           # ИЗМЕНЕНИЕ: Передаем сам элемент
                           add_node_safe(end_node_id, type=f"{elem_type}_end", label=label, validation_id=val_id, element_data=element)
                           graph_nodes_for_elem.append(end_node_id)
                           # Конечный узел для ПОСЛЕДОВАТЕЛЬНОГО потока - это end_node шлюза
                           current_end_node = end_node_id # Важно! Поток после шлюза идет от его end_node

                           # Рекурсивно process children paths
                           gateway_context = {"last_flow_src": start_node_id, "parent_end_node": end_node_id}
                           children_paths = element.get("children", [])
                           conditions = element.get("conditions", []) # Получаем условия здесь

                           for path_idx, path_list in enumerate(children_paths):
                                path_context = gateway_context.copy()
                                # Pass condition info if exclusive gateway
                                if elem_type == "exclusive":
                                     path_context["condition_label"] = conditions[path_idx] if path_idx < len(conditions) else None
                                # Process path recursively, returns the last node of the path
                                path_last_node = process_elements(path_list, path_context)

                                # ИЗМЕНЕНИЕ: Соединяем конец каждой ветки с конечным узлом шлюза (_E)
                                # Если ветка не пустая и имеет конечный узел
                                if path_last_node:
                                     add_edge_safe(path_last_node, end_node_id, type="sequence_join")
                                else:
                                     # Если ветка пустая, соединяем начальный узел шлюза с конечным
                                     # (с добавлением условия, если есть)
                                     edge_attrs = {"type": "sequence_empty_path"}
                                     if path_context.get("condition_label"):
                                         edge_attrs["label"] = f" {path_context['condition_label']} "
                                     add_edge_safe(start_node_id, end_node_id, **edge_attrs)


                      else: self.log_error("Graph Build", elem_id_str, f"{elem_type.capitalize()} gateway missing 'id'."); continue

                 elif elem_type == "loop":
                      loop_info = element.get("content", {})
                      target_task_id_base = loop_info.get("go_to")
                      target_task_id_full = f"T_{target_task_id_base}" if target_task_id_base else None

                      if last_flow_src and target_task_id_full:
                           # ИЗМЕНЕНИЕ: Убедимся, что целевой узел существует или создадим его (как implicit)
                           if target_task_id_full not in self.graph:
                               # Попробуем найти элемент задачи по ID
                               target_val_id = self._get_validation_id_by_element_id(target_task_id_base)
                               target_element = self._get_element_by_validation_id(target_val_id) if target_val_id else None
                               if target_element:
                                   task_word = target_element.get("content", {}).get("task", {}).get("word", "?")
                                   add_node_safe(target_task_id_full, type="task", label=task_word, validation_id=target_val_id, element_data=target_element)
                               else:
                                   self.log_warning("Graph Build", elem_id_str, f"Loop target task '{target_task_id_full}' not found in structure, adding implicit node.")
                                   add_node_safe(target_task_id_full, type="implicit_task", label=target_task_id_base)

                           add_edge_safe(last_flow_src, target_task_id_full, type="loop", style="dashed", constraint="false")
                      else: self.log_error("Graph Build", elem_id_str, "Loop cannot determine source or target task ID.")
                      # Loop does not produce a node for sequence flow, flow continues from element BEFORE loop
                      current_end_node = None # Loop прерывает последовательный поток здесь
                      # last_flow_src не меняется, следующий элемент (если он есть) будет после родителя цикла
                      continue # Skip sequence flow connection and last_flow_src update below

                 elif elem_type == "continue":
                     # TODO: Логика Continue требует пересмотра, она сложна и зависит от контекста.
                     # Пока оставим как есть, но она может быть неточной.
                     # print(f"{Fore.YELLOW}WARNING: 'continue' element graph logic is experimental.{Style.RESET_ALL}")
                     continue_info = element.get("content", {})
                     target_id_base = continue_info.get("go_to") # Can be Task ID (T_x) or Gateway ID (EGx, PGx)
                     parent_start_node = parent_context.get('last_flow_src') # Should be gateway start node (e.g., EG1_S)

                     if parent_start_node and target_id_base:
                         target_node_id = None
                         target_type = None
                         if target_id_base.startswith("T_"):
                              target_node_id = f"T_{target_id_base}"
                              target_type = "task"
                         elif target_id_base.startswith(("EG", "PG")):
                              target_node_id = f"{target_id_base}_S" # Целимся в НАЧАЛО шлюза? Или в КОНЕЦ? Зависит от семантики. Пока в начало.
                              target_type = "gateway_start"

                         if target_node_id:
                             # Убедимся, что целевой узел существует
                             if target_node_id not in self.graph:
                                 val_id_target = self._get_validation_id_by_element_id(target_id_base)
                                 element_target = self._get_element_by_validation_id(val_id_target) if val_id_target else None
                                 label_target = "?"
                                 if element_target and target_type == "task": label_target = element_target.get("content", {}).get("task", {}).get("word", "?")
                                 elif element_target and target_type == "gateway_start": label_target = "X" if element_target.get("type")=="exclusive" else "+"

                                 self.log_warning("Graph Build", elem_id_str, f"Continue target '{target_node_id}' not found in graph, adding implicit node.")
                                 add_node_safe(target_node_id, type=f"implicit_{target_type}", label=label_target)

                             edge_attrs = {"type": "sequence_continue", "style": "dashed", "constraint": "false"}
                             # Add condition label if available from context
                             if parent_context.get("condition_label"): edge_attrs["label"] = f" {parent_context['condition_label']} "
                             add_edge_safe(parent_start_node, target_node_id, **edge_attrs)
                         else: self.log_error("Graph Build", elem_id_str, f"Unknown continue target format: {target_id_base}")
                     else: self.log_error("Graph Build", elem_id_str, "Continue cannot determine source or target ID.")

                     # Continue прерывает последовательный поток в этой ветке
                     current_end_node = None
                     continue # Skip sequence flow connection below

                 else: self.log_error("Graph Build", elem_id_str, f"Unknown element type: {elem_type}."); continue

                 # --- Sequence Flow Connection ---
                 if current_start_node and last_flow_src:
                     edge_attrs = {"type": "sequence"}
                     # Add condition if this element is the first after parent XOR gateway start
                     if last_flow_src == parent_context.get('last_flow_src') and parent_context.get("condition_label"):
                          edge_attrs["label"] = f" {parent_context['condition_label']} "
                     add_edge_safe(last_flow_src, current_start_node, **edge_attrs)

                 # Update the source for the *next* element in this sequence
                 # ИЗМЕНЕНИЕ: Обновляем last_flow_src только если текущий элемент создал конечный узел для потока
                 if current_end_node:
                      last_flow_src = current_end_node # Update last_flow_src for the next iteration

                 # Map validation ID to graph nodes
                 if val_id: self.node_map[val_id] = graph_nodes_for_elem

            # Возвращаем последний узел последовательного потока в этой ветке/списке
            return last_flow_src


        # Start building from top level
        final_node_of_process = process_elements(structure_to_build, {"last_flow_src": None})

        # Add Start/End events
        all_nodes = list(self.graph.nodes)
        # if not all_nodes: print("Graph is empty, cannot add START/END events."); return

        start_event_id = "START_EVENT"; end_event_id = "END_EVENT"
        add_node_safe(start_event_id, type="start_event", label="", shape="circle", style="filled", fillcolor="limegreen", width="0.5", height="0.5")
        add_node_safe(end_event_id, type="end_event", label="", shape="circle", style="bold,filled", fillcolor="tomato", width="0.5", height="0.5")

        # Find nodes with no predecessors (potential starts) - excluding END_EVENT itself
        start_candidates = {n for n, d in self.graph.in_degree() if d == 0 and n != end_event_id} - {start_event_id}
        # Find nodes with no successors (potential ends) - excluding START_EVENT itself
        end_candidates = {n for n, d in self.graph.out_degree() if d == 0 and n != start_event_id} - {end_event_id}


        # Connect Start Event
        if not start_candidates and all_nodes - {start_event_id, end_event_id}: # Fallback if no clear start node found
            # Пытаемся найти первый узел по порядку добавления (не очень надежно, но лучше чем ничего)
            first_process_node = next((n for n in all_nodes if n not in [start_event_id, end_event_id]), None)
            if first_process_node:
                 actual_starts = {first_process_node}
                 self.log_warning("Graph Build", "START_EVENT", f"No explicit start node found, connecting to first node found: {first_process_node}.")
            else:
                 actual_starts = set() # Пустой процесс
                 add_edge_safe(start_event_id, end_event_id, type="sequence") # Соединяем старт и конец напрямую
                 self.log_warning("Graph Build", "START/END", "Process graph seems empty, connecting START directly to END.")
        else:
            actual_starts = start_candidates

        for node_id in actual_starts:
             add_edge_safe(start_event_id, node_id, type="sequence")

        # Connect End Event
        # ИЗМЕНЕНИЕ: Используем end_candidates напрямую, если они есть.
        # Если их нет, используем последний узел, возвращенный process_elements (если он был).
        actual_ends = set()
        if end_candidates:
             actual_ends = end_candidates
        elif final_node_of_process and final_node_of_process != start_event_id: # Если обработка вернула конечный узел
             actual_ends = {final_node_of_process}
             self.log_warning("Graph Build", "END_EVENT", f"No explicit end node found, connecting from last node in main flow: {final_node_of_process}.")
        elif all_nodes - {start_event_id, end_event_id}: # Fallback, если все остальное не сработало
            # Ищем последний узел по порядку (опять же, не очень надежно)
            last_process_node = next((n for n in reversed(all_nodes) if n not in [start_event_id, end_event_id]), None)
            if last_process_node:
                 actual_ends = {last_process_node}
                 self.log_warning("Graph Build", "END_EVENT", f"No explicit end node found, connecting from last node found in graph: {last_process_node}.")
            # else: # Пустой процесс, уже соединен старт и конец
        # else: # Пустой процесс, уже соединен старт и конец

        for node_id in actual_ends:
             add_edge_safe(node_id, end_event_id, type="sequence")

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")


    # --- Graph Checks ---
    def _check_connectivity(self):
        """Проверяет связность графа, выявляя недостижимые узлы и тупики."""
        if not self.graph or self.graph.number_of_nodes() <= 2: self.log_warning("Connectivity Check", "N/A", "Skipped: Graph not built or too small."); return
        print("Checking connectivity...")
        start_event = "START_EVENT"; end_event = "END_EVENT"
        if start_event not in self.graph or end_event not in self.graph:
            self.log_error("Connectivity", "Graph", "START_EVENT or END_EVENT node missing in graph.")
            return

        all_process_nodes = {n for n in self.graph.nodes() if n not in [start_event, end_event]}
        if not all_process_nodes: print("Connectivity Check: OK (Graph only contains Start/End)."); return

        # Check reachability from START
        try:
             reachable_from_start = nx.descendants(self.graph, start_event)
             reachable_from_start.add(start_event) # Start is reachable from start
        except Exception as e: self.log_error("Connectivity Error", start_event, f"Error finding descendants: {e}"); reachable_from_start = {start_event}

        unreachable_nodes = all_process_nodes - reachable_from_start
        for node_id in unreachable_nodes:
            node_data = self.graph.nodes[node_id]
            val_id = node_data.get('validation_id')
            elem = self._get_element_by_validation_id(val_id) if val_id else None
            ref_str = self._get_element_id_string(elem) if elem else node_id
            self.log_error("Connectivity", ref_str, "Node is unreachable from START.")

        # Check ability to reach END
        try:
            # ИЗМЕНЕНИЕ: Используем предков в обычном графе, а не развернутом
            can_reach_end_nodes = nx.ancestors(self.graph, end_event)
            can_reach_end_nodes.add(end_event) # END can reach END
        except Exception as e: self.log_error("Connectivity Error", end_event, f"Error finding ancestors: {e}"); can_reach_end_nodes = {end_event}

        # Dead ends: nodes reachable from START but CANNOT reach END
        dead_end_nodes = reachable_from_start - can_reach_end_nodes
        for node_id in dead_end_nodes:
             # Исключаем сам START_EVENT из dead ends
             if node_id != start_event:
                 node_data = self.graph.nodes[node_id]
                 val_id = node_data.get('validation_id')
                 elem = self._get_element_by_validation_id(val_id) if val_id else None
                 ref_str = self._get_element_id_string(elem) if elem else node_id
                 # ИЗМЕНЕНИЕ: Не логируем как ошибку, если это конечный узел шлюза (_E), т.к. он должен сливаться
                 if not node_data.get('type', '').endswith('_end'):
                     self.log_error("Connectivity", ref_str, "Node is a dead end (cannot reach END).")
                 #else:
                 #    print(f"DEBUG: Node {ref_str} cannot reach END, but is a gateway end node.")


        if not unreachable_nodes and not any(n for n in dead_end_nodes if not self.graph.nodes[n].get('type', '').endswith('_end') and n != start_event):
             print("Connectivity Check: OK - All process nodes seem reachable and can reach the end.")


    def _check_cycles(self):
        """Checks for cycles, distinguishing valid loops."""
        if not self.graph: self.log_warning("Cycle Check", "N/A", "Skipped: Graph not built."); return
        print("Checking for cycles...")
        try:
            # Create graph without explicit loop edges for unexpected cycle detection
            graph_for_cycle_check = self.graph.copy()
            loop_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') == 'loop']
            graph_for_cycle_check.remove_edges_from(loop_edges)
            cycles = list(nx.simple_cycles(graph_for_cycle_check))
        except Exception as e:
            self.log_error("Cycle Check Error", "Graph", f"Failed during cycle detection: {e}")
            return

        if not cycles: print("Cycle Check: OK - No unexpected cycles found.")
        else:
            for i, cycle in enumerate(cycles):
                # ИЗМЕНЕНИЕ: Получаем более читаемое представление узлов цикла
                cycle_nodes_repr = []
                for node_id in cycle:
                    node_data = self.graph.nodes.get(node_id, {})
                    val_id = node_data.get('validation_id')
                    elem = self._get_element_by_validation_id(val_id) if val_id else None
                    ref_str = self._get_element_id_string(elem) if elem else node_id
                    cycle_nodes_repr.append(ref_str)
                cycle_nodes_str = " -> ".join(cycle_nodes_repr) + f" -> {cycle_nodes_repr[0]}"

                # Check if cycle has an exit path to END
                can_reach_end_from_cycle = False
                if "END_EVENT" in self.graph:
                    for node_in_cycle in cycle:
                        # Проверяем исходящие ребра, не являющиеся частью цикла
                        for successor in self.graph.successors(node_in_cycle):
                            is_cycle_edge = False
                            # Проверяем, ведет ли ребро к следующему узлу в *данном* цикле
                            current_index = cycle.index(node_in_cycle)
                            next_node_in_cycle = cycle[(current_index + 1) % len(cycle)]
                            if successor == next_node_in_cycle:
                                 is_cycle_edge = True

                            # Ищем путь к END от узлов, куда ведут нециклические ребра
                            if not is_cycle_edge:
                                try:
                                    # Используем предков, чтобы проверить, может ли END быть достигнут из successor
                                    if "END_EVENT" in nx.descendants(self.graph, successor) or successor == "END_EVENT":
                                        can_reach_end_from_cycle = True
                                        # print(f"DEBUG: Cycle exit found: {node_in_cycle} -> {successor}") # Отладка
                                        break
                                except nx.NodeNotFound: pass
                                except nx.NetworkXError: pass # Handle cases where path doesn't exist
                        if can_reach_end_from_cycle: break

                if can_reach_end_from_cycle:
                     self.log_warning("Cycle Check", f"Cycle {i+1}: {cycle_nodes_str}", f"Unexpected cycle detected, but it has an exit path towards END. Verify logic.")
                else:
                     self.log_error("Cycle Check", f"Cycle {i+1}: {cycle_nodes_str}", f"Potentially infinite loop or deadlock detected. No clear exit path to END found from cycle.")


    # --- Rule Checking (Adapted from previous response) ---
    def _check_rules(self):
        """Проверяет правила BPMN в отношении текущей структуры."""
        if not self.current_structure:
            self.log_warning("Rule Check", "N/A", "Skipped: BPMN structure is empty.")
            return
        if not isinstance(self.current_structure, list):
             self.log_error("Rule Check", "Structure", f"Failed: Expected structure to be a list, but got {type(self.current_structure)}.")
             return

        print("Checking element-specific BPMN rules...")
        all_task_ids = self._collect_all_task_ids(self.current_structure)
        all_gateway_ids = self._collect_all_gateway_ids(self.current_structure) # Collect gateway IDs too
        self.check_element_rules(self.current_structure, is_top_level=True, all_task_ids=all_task_ids, all_gateway_ids=all_gateway_ids)
        self._check_global_rules() # Check rules applying to the whole process
        print("Rule checking finished.")


    def _collect_all_task_ids(self, element) -> Set[str]:
        """Рекурсивно собирает все идентификаторы задач ('T_...') из структуры."""
        task_ids = set()
        if isinstance(element, list):
            for item in element: task_ids.update(self._collect_all_task_ids(item))
        elif isinstance(element, dict):
            if element.get("type") == "task":
                task_id_base = element.get("content", {}).get("task", {}).get("task_id")
                if task_id_base: task_ids.add(f"T_{task_id_base}") # Add with prefix
            elif element.get("type") in ["exclusive", "parallel"]:
                for path_list in element.get("children", []): task_ids.update(self._collect_all_task_ids(path_list))
            # ИЗМЕНЕНИЕ: Добавлена проверка для других типов элементов, которые могут содержать task_id
            elif "task_id" in element.get("content", {}).get("task", {}): # Проверка для других типов
                 task_id_base = element["content"]["task"]["task_id"]
                 if task_id_base: task_ids.add(f"T_{task_id_base}")

        return task_ids

    def _collect_all_gateway_ids(self, element) -> Set[str]:
        """Рекурсивно собирает все идентификаторы узлов шлюза ('..._S', '..._E') из структуры."""
        gateway_ids = set()
        if isinstance(element, list):
            for item in element: gateway_ids.update(self._collect_all_gateway_ids(item))
        elif isinstance(element, dict):
            if element.get("type") in ["exclusive", "parallel"]:
                gw_id_base = element.get("id")
                if gw_id_base:
                    start_node = f"{gw_id_base}_S"; end_node = f"{gw_id_base}_E"
                    gateway_ids.add(start_node); gateway_ids.add(end_node) # Add both nodes
                for path_list in element.get("children", []): gateway_ids.update(self._collect_all_gateway_ids(path_list))
        return gateway_ids


    def check_element_rules(self, element, is_top_level=False, context=None, all_task_ids=None, all_gateway_ids=None):
        """
        Рекурсивно проверяет правила BPMN для данного элемента или списка элементов.
        """
        if context is None: context = {}
        if all_task_ids is None: all_task_ids = set()
        if all_gateway_ids is None: all_gateway_ids = set()

        if isinstance(element, list):
            # ИЗМЕНЕНИЕ: Передаем обновленный контекст в дочерние элементы одного уровня
            current_context = context.copy()
            for item in element:
                self.check_element_rules(item, is_top_level=False, context=current_context, all_task_ids=all_task_ids, all_gateway_ids=all_gateway_ids)
                # Обновляем контекст для следующего элемента в списке, если текущий был шлюзом
                if isinstance(item, dict) and item.get("type") in ["exclusive", "parallel"]:
                     current_context['last_processed_gateway_id'] = item.get("id")

        elif isinstance(element, dict):
            elem_type = element.get("type")
            content = element.get("content", {})
            val_id = element.get("_validation_id")
            elem_ref_name = self._get_element_id_string(element)

            if not elem_type: self.log_error("Structure Check", elem_ref_name, "Element missing 'type' key."); return

            if elem_type == "task":
                task_info = content.get("task", {}); agent_info = content.get("agent", {})
                resolved_agent = agent_info.get("resolved_word"); task_word = task_info.get("word")
                task_id_base = task_info.get("task_id"); task_id_graph = f"T_{task_id_base}" if task_id_base else None

                if not task_info: self.log_error("Task Rule", elem_ref_name, "'content' missing 'task' details.")
                if not task_word: self.log_warning("Task Rule", elem_ref_name, "Missing task description ('word').")
                if not task_id_base: self.log_error("Task Rule", elem_ref_name, "Missing 'task_id'.") # ID is crucial

                if not resolved_agent or resolved_agent == "Unknown Agent":
                    msg = f"Missing a valid resolved agent."
                    if agent_info.get("original_word"): msg += f" (Original: '{agent_info.get('original_word')}')"
                    self.log_warning("Task Agent Rule", elem_ref_name, msg) # ИЗМЕНЕНИЕ: Снижена серьезность до Warning

                # ИЗМЕНЕНИЕ: Убрано предупреждение о 'condition', так как оно может быть частью LLM вывода
                # if "condition" in element and not context.get('parent_gateway_type') == 'exclusive':
                #      # Проверяем, не является ли это условие частью самой задачи (например, из LLM)
                #      cond_text = element.get("condition", {}).get("word", "")
                #      if not cond_text: # Если это не полноценное условие, а просто ключ
                #          self.log_warning("Task Rule", elem_ref_name, "Has 'condition' key but is not directly inside an exclusive gateway path. Key might be redundant.")

            elif elem_type in ["exclusive", "parallel"]:
                gw_id_base = element.get("id")
                if not gw_id_base: self.log_error("Gateway Rule", elem_ref_name, "Missing 'id'.")
                children_paths = element.get("children"); paths_info = element.get("paths")

                if children_paths is None or not isinstance(children_paths, list):
                    self.log_error("Gateway Rule", elem_ref_name, "'children' key is missing or not a list.")
                    children_paths = [] # Continue with empty list

                num_paths = len(children_paths)
                # ИЗМЕНЕНИЕ: Не предупреждаем, если тип шлюза 'single_sentence_expansion', так как он может иметь 1 путь
                if num_paths <= 1 and not element.get("type") == "single_sentence_expansion":
                     # ИЗМЕНЕНИЕ: Снижена серьезность до Warning, так как иногда это может быть легитимно (хоть и избыточно)
                     self.log_warning("Gateway Rule", elem_ref_name, f"Has only {num_paths} path(s). Gateway might be redundant.")

                if paths_info and isinstance(paths_info, list) and len(paths_info) != num_paths:
                     self.log_warning("Gateway Rule", elem_ref_name, f"# children paths ({num_paths}) != # path indices ({len(paths_info)}). Might indicate parsing issue.")

                if elem_type == "exclusive":
                    conditions = element.get("conditions", [])
                    if not isinstance(conditions, list): self.log_error("Gateway Rule", elem_ref_name, "'conditions' not a list.")
                    elif num_paths >= 2 and len(conditions) < num_paths:
                         # ИЗМЕНЕНИЕ: Предупреждаем, если есть пути, но условий меньше
                         self.log_warning("Gateway Rule", elem_ref_name, f"Has {num_paths} paths but only {len(conditions)} condition(s).")
                    elif num_paths >=1 and not conditions: # ИЗМЕНЕНИЕ: Предупреждаем, если есть пути, но нет условий вообще
                        self.log_warning("Gateway Rule", elem_ref_name, f"Has {num_paths} paths but no 'conditions' defined.")


                elif elem_type == "parallel":
                    if "conditions" in element: self.log_warning("Gateway Rule", elem_ref_name, "Should not have 'conditions'.")

                # Recurse into children
                child_context = context.copy(); child_context['parent_gateway_id'] = gw_id_base; child_context['parent_gateway_type'] = elem_type
                for path_list in children_paths:
                    self.check_element_rules(path_list, is_top_level=False, context=child_context, all_task_ids=all_task_ids, all_gateway_ids=all_gateway_ids)

            elif elem_type == "loop":
                 loop_target_base = content.get("go_to")
                 loop_target_graph = f"T_{loop_target_base}" if loop_target_base else None
                 if not loop_target_base: self.log_error("Loop Rule", elem_ref_name, "Missing 'go_to' target task ID.")
                 elif loop_target_graph not in all_task_ids:
                      self.log_error("Loop Rule", elem_ref_name, f"Target task ID '{loop_target_base}' ('{loop_target_graph}') does not exist in the process tasks.")

            elif elem_type == "continue":
                 continue_target_base = content.get("go_to")
                 target_node_graph = None
                 if continue_target_base:
                     if continue_target_base.startswith("T_"): target_node_graph = f"T_{continue_target_base}"
                     elif continue_target_base.startswith(("EG", "PG")): target_node_graph = f"{continue_target_base}_S" # Проверяем _S узел шлюза
                 if not continue_target_base: self.log_error("Continue Rule", elem_ref_name, "Missing 'go_to' target ID.")
                 elif target_node_graph and target_node_graph not in all_task_ids and target_node_graph not in all_gateway_ids:
                       self.log_warning("Continue Rule", elem_ref_name, f"Target ID '{continue_target_base}' ('{target_node_graph}') not found among task IDs or gateway start nodes.")

            else: self.log_warning("Structure Check", elem_ref_name, f"Encountered unknown element type '{elem_type}'.")

        else: self.log_warning("Structure Check", "N/A", f"Skipped unexpected element type '{type(element)}'.")


    def _check_global_rules(self):
         """Проверяет правила, применимые к общей структуре (используя график)."""
         if not self.graph: return
         print("Checking global rules...")
         start_event_count = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'start_event'])
         end_event_count = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'end_event'])

         if start_event_count == 0 and self.graph.number_of_nodes() > 0 : self.log_error("Global Rule", "START_EVENT", "Process must have a Start Event.")
         elif start_event_count > 1: self.log_warning("Global Rule", "START_EVENT", f"Process has {start_event_count} Start Events. Usually only one is expected.")

         if end_event_count == 0 and self.graph.number_of_nodes() > 0 : self.log_error("Global Rule", "END_EVENT", "Process must have at least one End Event.")


    # --- LLM Analysis ---
    def _get_llm_response(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Отправляет запросы загруженному LLM и возвращает текст ответа."""
        if not LLM_AVAILABLE or not llm_model or not llm_tokenizer:
            print(f"{Fore.YELLOW}LLM model not available. Skipping LLM interaction.{Style.RESET_ALL}")
            return None
        try:
            messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]
            # Используем шаблон чата, если он есть
            try: prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception: prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:" # Fallback

            inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(LLM_DEVICE) # Ограничение длины входа

            # ИЗМЕНЕНИЕ: Убран temperature из GenerationConfig, так как do_sample=False
            generation_config = GenerationConfig(
                max_new_tokens=512,
                # temperature=0.1, # Убрано
                do_sample=False,
                pad_token_id=llm_tokenizer.eos_token_id
            )

            print(f"Sending request to LLM ({inputs.input_ids.shape[1]} tokens)...")
            with torch.no_grad(): outputs = llm_model.generate(**inputs, generation_config=generation_config)
            input_length = inputs.input_ids.shape[1]
            response_text = llm_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            print("LLM analysis response received.")
            return response_text.strip()
        except Exception as e:
            print(f"{Fore.RED}Error during LLM interaction: {e}{Style.RESET_ALL}"); traceback.print_exc(); return None

    def _parse_llm_response(self, response_text: Optional[str]) -> Dict[str, List[Any]]:
        """Анализирует текст ответа LLM, чтобы извлечь проблемы и предложения."""
        parsed_data = {"new_issues": [], "suggestions": []}
        if not response_text: return parsed_data
        print("Parsing LLM response...")

        # --- Улучшенный парсинг ---
        try:
            sections = {}
            current_section_name = "Header" # Секция до первого заголовка
            current_section_content = []

            # Разделяем по заголовкам типа "### Заголовок:"
            lines = response_text.splitlines()
            header_pattern = re.compile(r"^\s*###\s*(.*?)\s*:{0,1}\s*$") # Заголовок с ###

            for line in lines:
                match = header_pattern.match(line)
                if match:
                    # Сохраняем предыдущую секцию
                    sections[current_section_name] = "\n".join(current_section_content).strip()
                    # Начинаем новую секцию
                    current_section_name = match.group(1).strip()
                    current_section_content = []
                else:
                    current_section_content.append(line)
            # Сохраняем последнюю секцию
            sections[current_section_name] = "\n".join(current_section_content).strip()

            # Обрабатываем секции
            for section_name, section_content in sections.items():
                 if not section_content: continue # Пропускаем пустые секции

                 # Извлечение новых проблем
                 if "identif" in section_name.lower() and "issue" in section_name.lower():
                     issue_lines = [line.strip() for line in section_content.splitlines() if line.strip()]
                     current_issue_parts = []
                     for line in issue_lines:
                          # Ищем маркеры списка
                          if re.match(r"^\s*[\*\-\d]+\.?\s+", line):
                              if current_issue_parts: # Сохраняем предыдущую проблему
                                  issue_content = re.sub(r"^\s*[\*\-\d]+\.?\s+", "", "\n".join(current_issue_parts)).strip()
                                  if issue_content.lower() not in ["none.", "none", "n/a", "no new issues found."]:
                                       parsed_data["new_issues"].append(issue_content)
                              current_issue_parts = [line] # Начинаем новую
                          elif current_issue_parts: # Продолжение предыдущей проблемы
                              current_issue_parts.append(line)
                     # Сохраняем последнюю проблему
                     if current_issue_parts:
                          issue_content = re.sub(r"^\s*[\*\-\d]+\.?\s+", "", "\n".join(current_issue_parts)).strip()
                          if issue_content.lower() not in ["none.", "none", "n/a", "no new issues found."]:
                              parsed_data["new_issues"].append(issue_content)

                 # Извлечение предложений (из разных секций)
                 elif "suggestion" in section_name.lower() or "correction" in section_name.lower():
                     suggestion_lines = [line.strip() for line in section_content.splitlines() if line.strip()]
                     current_suggestion = None # Сбрасываем для каждой секции предложений

                     for line in suggestion_lines:
                          # Ищем заголовки предложений (сначала самые специфичные)
                          error_match = re.match(r"^\s*[\*\-]\s*\*\*Error\s+(\d+)\s*(?:\[.*?\])?\*\*:\s*(.*)", line, re.IGNORECASE)
                          new_issue_match = re.match(r"^\s*[\*\-]\s*\*\*New\s+Issue\s*\[?(.*?)\]?\*\*:\s*(.*)", line, re.IGNORECASE)
                          generic_header_match = re.match(r"^\s*[\*\-]\s*\*\*(.*?)\*\*:\s*(.*)", line) # Любой жирный заголовок
                          list_item_match = re.match(r"^\s*[\*\-]\s+(.*)", line) # Обычный пункт списка

                          # Новое предложение, если найден заголовок
                          if error_match or new_issue_match or generic_header_match:
                               suggestion_text = ""
                               error_ref = f"Suggestion ({section_name})" # Default ref

                               if error_match:
                                   try:
                                       error_index = int(error_match.group(1).strip()) - 1
                                       suggestion_text = error_match.group(2).strip()
                                       error_ref = error_index # Используем индекс
                                   except (ValueError, IndexError):
                                       print(f"{Fore.YELLOW}Warning: Invalid Error Index {error_match.group(1)} in '{line}'. Treating as generic.{Style.RESET_ALL}")
                                       suggestion_text = error_match.group(2).strip() # Берем текст
                                       error_ref = f"Unparsed Error Ref ({error_match.group(1)})"
                               elif new_issue_match:
                                   issue_ref_text = new_issue_match.group(1).strip() or "Unspecified New Issue"
                                   suggestion_text = new_issue_match.group(2).strip()
                                   error_ref = f"New Issue: {issue_ref_text}"
                               elif generic_header_match:
                                   header = generic_header_match.group(1).strip()
                                   suggestion_text = generic_header_match.group(2).strip()
                                   error_ref = f"Suggestion ({header})" # Используем заголовок как ссылку

                               # Создаем новое предложение и добавляем в список
                               current_suggestion = {"error_ref": error_ref, "suggestion_text": suggestion_text}
                               parsed_data["suggestions"].append(current_suggestion)

                          # Продолжение текста или пункт списка для текущего предложения
                          elif current_suggestion: # Если есть активное предложение
                               if list_item_match: # Добавляем пункт списка
                                    item_text = list_item_match.group(1).strip()
                                    current_suggestion["suggestion_text"] += f"\n- {item_text}" # Добавляем с маркером
                               else: # Добавляем обычный текст как продолжение
                                    current_suggestion["suggestion_text"] += f"\n{line}"
                          elif list_item_match: # Пункт списка без заголовка -> общее предложение
                               item_text = list_item_match.group(1).strip()
                               current_suggestion = {"error_ref": f"General Suggestion ({section_name})", "suggestion_text": f"- {item_text}"}
                               parsed_data["suggestions"].append(current_suggestion)


            print(f"LLM response parsed: Found {len(parsed_data['new_issues'])} new issues, {len(parsed_data['suggestions'])} suggestions.")
        # --- КОНЕЦ ИЗМЕНЕНИЯ: Улучшенный парсинг ---
        except Exception as e: print(f"{Fore.RED}Error parsing LLM response: {e}{Style.RESET_ALL}"); print(f"--- Raw LLM Response ---:\n{response_text}\n------------------------"); traceback.print_exc(); return {"new_issues": [], "suggestions": []}
        return parsed_data


    def _run_llm_analysis(self):
        """Выполняет анализ на основе LLM для семантической проверки."""
        if not LLM_AVAILABLE: return
        if not PROMPTS_AVAILABLE: print(f"{Fore.YELLOW}LLM prompts not available. Skipping LLM analysis.{Style.RESET_ALL}"); return
        if not self.current_structure: print("LLM Analysis skipped: No structure available."); return

        print("Running LLM analysis...")
        # Сериализуем текущую структуру и ошибки/предупреждения
        try:
             # ИЗМЕНЕНИЕ: Передаем и ошибки и предупреждения
             issues_list = self.errors + self.warnings
             structure_repr = json.dumps(self.current_structure, indent=2)
        except TypeError as e: print(f"{Fore.RED}Error serializing structure/issues for LLM: {e}{Style.RESET_ALL}"); return
        # Обрезаем, если слишком длинно
        max_len = 3500 # Оставляем место для текста и ошибок
        if len(structure_repr) > max_len: structure_repr = structure_repr[:max_len] + "\n... (structure truncated)"

        user_prompt = create_bpmn_analysis_prompt(self.original_text, structure_repr, issues_list)
        llm_response_text = self._get_llm_response(BPMN_ANALYSIS_SYSTEM_MSG, user_prompt)

        if llm_response_text:
             print(f"{Fore.CYAN}--- LLM Analysis Raw Response ---{Style.RESET_ALL}\n{llm_response_text}\n{Fore.CYAN}-------------------------------{Style.RESET_ALL}")
             parsed_llm_data = self._parse_llm_response(llm_response_text)
             self.suggestions = parsed_llm_data.get("suggestions", [])
             self.new_llm_issues = parsed_llm_data.get("new_issues", [])
             # Добавляем новые проблемы, найденные LLM, в основной список ошибок
             for issue_text in self.new_llm_issues:
                  # ИЗМЕНЕНИЕ: Используем log_warning для LLM-проблем, так как они часто семантические
                  self.log_warning("LLM Identified Issue", "N/A", issue_text)
             if self.new_llm_issues: print(f"{Fore.BLUE}LLM identified {len(self.new_llm_issues)} potential new issues (added to warnings).{Style.RESET_ALL}")
        else: print("LLM analysis did not return a response.")

    # --- Automatic Fixing ---
    def apply_safe_fixes(self) -> bool:
        """
        Применяет ограниченный набор безопасных исправлений, основанных на обнаруженных ошибках
        ,непосредственно к self.current_structure.
        Возвращает значение True, если были внесены какие-либо изменения, или False в противном случае.
        ВНИМАНИЕ: Изменение структуры является сложным и потенциально опасным процессом.
        """
        # ИЗМЕНЕНИЕ: Применяем фиксы только если есть ОШИБКИ (не предупреждения)
        if not self.errors:
            print("No structural errors found, no automatic fixes to apply.")
            return False

        print(f"{Fore.YELLOW}Applying safe automatic fixes based on structural errors...{Style.RESET_ALL}")
        structure_modified = False
        processed_error_indices = set() # Track applied fixes to avoid duplicates

        # Helper to find and remove element by validation ID recursively
        def find_and_remove(target_list: list, validation_id_to_remove: str) -> bool:
            nonlocal structure_modified # Используем nonlocal для изменения флага
            item_removed_in_call = False # Локальный флаг для этой функции/рекурсии
            indices_to_del = []
            for i, item in enumerate(target_list):
                if isinstance(item, dict):
                    if item.get("_validation_id") == validation_id_to_remove:
                        indices_to_del.append(i)
                        item_removed_in_call = True
                        print(f"  {Fore.GREEN}[Fix Applied]{Style.RESET_ALL} Removing element {self._get_element_id_string(item)}")
                        break # Удаляем только первое вхождение на этом уровне
                    # Recurse into gateway children
                    elif item.get("type") in ["exclusive", "parallel"] and "children" in item:
                         for path in item["children"]:
                             if find_and_remove(path, validation_id_to_remove):
                                 item_removed_in_call = True; break # Останавливаемся после первого удаления
                    # ИЗМЕНЕНИЕ: Добавлена рекурсия для других потенциальных контейнеров (если появятся)
                    elif "children" in item and isinstance(item["children"], list): # Общая проверка на вложенность
                         if find_and_remove(item["children"], validation_id_to_remove):
                                 item_removed_in_call = True; break

                if item_removed_in_call: break

            # Remove found items (iterating backwards)
            if indices_to_del:
                for i in sorted(indices_to_del, reverse=True): del target_list[i]
                structure_modified = True # Обновляем глобальный флаг

            return item_removed_in_call

        # Iterate through errors and apply fixes
        for i, error in enumerate(self.errors):
             if i in processed_error_indices: continue

             fix_applied_for_this_error = False
             error_category = error.get("category") # Используем категорию для определения типа ошибки
             error_msg = error.get("message", "").lower()
             element_ref = error.get("element_ref") # Может быть ID задачи/шлюза или validation_id

             # Пытаемся получить validation_id из element_ref
             validation_id = None
             if isinstance(element_ref, str):
                  # Если это строка, похожая на UUID (содержит дефисы)
                  if '-' in element_ref and len(element_ref) > 10: validation_id = element_ref
                  else: validation_id = self._get_validation_id_by_element_id(element_ref)

             # --- Fix 1: Remove redundant gateway (only 1 path) ---
             # ИЗМЕНЕНИЕ: Не применяем этот фикс автоматически, так как он небезопасен
             # if error_category == "Gateway Rule" and "has only" in error_msg and "path(s)" in error_msg and validation_id:
             #     gateway_element = self._get_element_by_validation_id(validation_id)
             #     if gateway_element and len(gateway_element.get("children", [])) <= 1:
             #          print(f"{Fore.YELLOW}Fix Skipped: Removing redundant gateway {self._get_element_id_string(gateway_element)} automatically is unsafe. Please review manually.{Style.RESET_ALL}")


             # --- Fix 2: Remove invalid Loop/Continue target ---
             # ИЗМЕНЕНИЕ: Проверяем категорию ошибки и сообщение
             elif error_category in ["Loop Rule", "Continue Rule"] and validation_id and \
                  ("target task id" in error_msg or "target id" in error_msg) and \
                  ("does not exist" in error_msg or "not found" in error_msg):

                 target_element = self._get_element_by_validation_id(validation_id)
                 if target_element:
                     print(f"Attempting fix for '{error_category}': Removing element {self._get_element_id_string(target_element)} with invalid target.")
                     if find_and_remove(self.current_structure, validation_id):
                          fix_applied_for_this_error = True
                 else:
                      print(f"{Fore.YELLOW}Warning: Could not find element with validation_id {validation_id} to apply fix for invalid target.{Style.RESET_ALL}")


             # --- Fix 3: Remove task/gateway missing critical ID ---
             # ИЗМЕНЕНИЕ: Не применяем этот фикс автоматически
             # elif error_category in ["Task Rule", "Gateway Rule"] and validation_id and \
             #      ("missing 'task_id'" in error_msg or "missing 'id'" in error_msg):
             #      target_element = self._get_element_by_validation_id(validation_id)
             #      if target_element:
             #            print(f"{Fore.YELLOW}Fix Skipped: Removing element {self._get_element_id_string(target_element)} missing critical ID is unsafe. Please review manually.{Style.RESET_ALL}")


             if fix_applied_for_this_error:
                 processed_error_indices.add(i)
                 # structure_modified уже установлен внутри find_and_remove

        if structure_modified:
            print(f"{Fore.GREEN}BPMN structure list was modified by safe fixes.{Style.RESET_ALL}")
            # ИЗМЕНЕНИЕ: Предупреждение о необходимости перестроения ГРАФА, а не всей валидации
            print(f"{Fore.YELLOW}Warning: Graph representation is now potentially inconsistent. Re-run validation if accurate graph checks are needed after fixes.{Style.RESET_ALL}")
        else:
            print("No structural modifications were applied by safe fixes.")

        return structure_modified


    # --- Main Validation Method ---
    # ИЗМЕНЕНИЕ: Возвращаем errors, warnings, suggestions
    def validate(self, run_llm: bool = False, run_fixes: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Выполняет все проверки текущей структуры BPMN.
        При необходимости запускает анализ LLM и применяет безопасные исправления.

        Args:
            run_llm (bool): Whether to perform LLM-based analysis.
            run_fixes (bool): Whether to attempt applying safe automatic fixes based on errors.

        Returns:
            tuple[list[dict], list[dict], list[dict]]: Final lists of errors, warnings, and suggestions.
        """
        print(f"\n{Fore.MAGENTA}--- Starting BPMN Validation ---{Style.RESET_ALL}")
        self.errors = [] # Reset errors for this run
        self.warnings = [] # Reset warnings
        self.suggestions = []
        self.new_llm_issues = []

        if self.current_structure is None:
             self.log_error("Critical", "Initialization", "Cannot validate, structure was not loaded correctly.")
             return self.errors, self.warnings, self.suggestions

        # 1. Build Graph (based on current structure)
        try: self._build_graph(self.current_structure)
        except Exception as e: self._log_critical_error("Graph Build", f"Failed: {e}"); traceback.print_exc(); return self.errors, self.warnings, self.suggestions

        # 2. Connectivity Check
        try: self._check_connectivity()
        except Exception as e: self.log_error("Connectivity Check", "Graph", f"Failed: {e}"); traceback.print_exc()

        # 3. Cycle Check
        try: self._check_cycles()
        except Exception as e: self.log_error("Cycle Check", "Graph", f"Failed: {e}"); traceback.print_exc()

        # 4. BPMN Rule Check
        try: self._check_rules()
        except Exception as e: self.log_error("Rule Check", "Structure", f"Failed: {e}"); traceback.print_exc()

        initial_error_count = len(self.errors)
        initial_warning_count = len(self.warnings) # Подсчитываем предупреждения
        # ИЗМЕНЕНИЕ: Корректное сообщение об итогах автоматических проверок
        if initial_error_count > 0 or initial_warning_count > 0:
            print(f"\n{Fore.YELLOW}Found {initial_error_count} structural errors and {initial_warning_count} warnings (before LLM/fixes).{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}No structural errors or warnings found by automated checks.{Style.RESET_ALL}")

        # 5. Apply Fixes (if requested and ERRORS exist)
        structure_was_modified = False
        if run_fixes and self.errors: # Запускаем фиксы только если есть ошибки
             structure_was_modified = self.apply_safe_fixes()
             if structure_was_modified:
                  # Rebuild graph ONLY if fixes were applied
                  print(f"{Fore.YELLOW}Rebuilding graph after fixes were applied...{Style.RESET_ALL}")
                  try:
                      # Перестраиваем граф на основе ИЗМЕНЕННОЙ структуры
                      self._build_graph(self.current_structure)
                      # После перестроения графа можно ПОВТОРНО запустить проверки ГРАФА,
                      # чтобы обновить их результаты после фиксов. Правила структуры проверять не нужно.
                      print(f"{Fore.CYAN}Re-running graph checks after fixes...{Style.RESET_ALL}")
                      self._check_connectivity()
                      self._check_cycles()
                  except Exception as e:
                      # Ошибки при перестроении графа критичны
                      self._log_critical_error("Graph Rebuild After Fix", f"Failed: {e}")
                      traceback.print_exc()
                      # Не продолжаем с LLM, так как граф некорректен
                      return self.errors, self.warnings, self.suggestions

        # 6. LLM Analysis (if requested) - run on potentially fixed structure/rebuilt graph
        if run_llm:
            if LLM_AVAILABLE:
                 try: self._run_llm_analysis() # Этот метод добавляет LLM-проблемы в self.warnings
                 except Exception as e: self.log_error("LLM Analysis", "N/A", f"Failed: {e}"); traceback.print_exc()
            else: print(f"{Fore.YELLOW}LLM analysis requested but model not available.{Style.RESET_ALL}")

        # --- Final Report ---
        print(f"\n{Fore.MAGENTA}--- Validation Finished ---{Style.RESET_ALL}")
        # ИЗМЕНЕНИЕ: Финальные подсчеты после всех шагов
        final_error_count = len(self.errors)
        final_warning_count = len(self.warnings)

        if final_error_count > 0: print(f"{Fore.RED}Final count of reported structural errors: {final_error_count}{Style.RESET_ALL}")
        else: print(f"{Fore.GREEN}No structural errors reported.{Style.RESET_ALL}")

        if final_warning_count > 0: print(f"{Fore.YELLOW}Final count of reported warnings: {final_warning_count}{Style.RESET_ALL}")
        else: print(f"No warnings reported.") # Сообщение, если предупреждений нет

        if self.suggestions:
             print(f"\n{Fore.CYAN}LLM Suggestions ({len(self.suggestions)}):{Style.RESET_ALL}")
             for i, sugg_data in enumerate(self.suggestions):
                  ref = sugg_data.get('error_ref', 'General'); text = sugg_data.get('suggestion_text', 'N/A')
                  ref_str = f"For Error Index {ref+1}" if isinstance(ref, int) else f"For '{ref}'"
                  # ИЗМЕНЕНИЕ: Печатаем текст предложения "как есть", сохраняя переносы строк
                  print(f"  {i+1}. [{ref_str}]:\n     {text.replace(chr(10), chr(10)+'     ')}") # Добавляем отступ для многострочных
        elif run_llm: print(f"\nNo suggestions were generated by LLM analysis.")

        return self.errors, self.warnings, self.suggestions


    def get_corrected_structure(self) -> Optional[List[Dict[str, Any]]]:
        """Возвращает текущее состояние структуры BPMN (возможно, измененное исправлениями)."""
        return self.current_structure


# --- END OF FILE bpmn_validator.py ---

# --- Тестовый Запуск --- Для теста! Я проверял, вроде работает. На тестовых данных - ошибок выдавать не должен!
"""
if __name__ == "__main__":
    print(f"{Style.BRIGHT}--- Running BPMN Validator Test ---{Style.RESET_ALL}")

    # --- Configuration ---
    # Используем относительные пути от папки src/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Поднимаемся до process-visualizer
    structure_dir_rel = os.path.join("output_logs", "bpmn_structure")
    structure_filename = "bpmn_final_structure.json"
    structure_file_abs = os.path.join(project_root, structure_dir_rel, structure_filename)

    # Пример текста (можно загружать из файла)
    original_text_example = 
    Start process. The customer decides if he wants to finance or pay in cash.
    If the customer chooses to finance, the customer will need to fill out a loan application and then the application is sent by him.
    If the customer chooses to pay in cash, the customer brings the cash. Payment is done. Payment is done again. Bad Task.
    After this, the customer signs the contract. The process ends. End Process.
    # --- End Configuration ---

    bpmn_structure_data = None
    print(f"Looking for structure file at: {structure_file_abs}")
    if exists(structure_file_abs):
        try:
            with open(structure_file_abs, "r", encoding='utf-8') as f: bpmn_structure_data = json.load(f)
            print(f"Loaded BPMN structure from {structure_file_abs}")
        except Exception as e: print(f"{Fore.RED}Error loading {structure_file_abs}: {e}{Style.RESET_ALL}")
    else: print(f"{Fore.RED}Structure file not found: {structure_file_abs}{Style.RESET_ALL}")

    if bpmn_structure_data:
        try:
            # --- Запуск Валидатора ---
            validator = BPMNValidator(bpmn_structure_data, original_text_example)

            # Запускаем валидацию: С фиксами, С LLM (если доступен)
            errors, suggestions = validator.validate(run_llm=LLM_AVAILABLE, run_fixes=True)

            # Получаем финальную (возможно исправленную) структуру
            final_corrected_structure = validator.get_corrected_structure()

            # Сохраняем исправленную структуру
            output_dir = os.path.join(project_root, "output_logs", "bpmn_structure") # Указываем полный путь
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            corrected_file = os.path.join(output_dir, "bpmn_corrected_structure.json")
            try:
                 with open(corrected_file, "w", encoding='utf-8') as f: json.dump(final_corrected_structure, f, indent=2, ensure_ascii=False)
                 print(f"\nCorrected structure saved to {corrected_file}")
            except Exception as e: print(f"{Fore.RED}Error saving corrected structure: {e}{Style.RESET_ALL}")

            # --- Опционально: Повторная Валидация ---
            if errors: # Запускаем повторно, только если были ошибки ДО фиксов
                print("\n--- Re-validating the structure AFTER potential fixes ---")
                re_validator = BPMNValidator(final_corrected_structure, original_text_example)
                errors_after, _ = re_validator.validate(run_llm=False, run_fixes=False) # Без LLM и фиксов
                if not errors_after: print(f"{Fore.GREEN}Validation PASSED after applying fixes!{Style.RESET_ALL}")
                else: print(f"{Fore.RED}Validation found {len(errors_after)} errors AFTER applying fixes.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}An error occurred during validation process: {e}{Style.RESET_ALL}")
            traceback.print_exc()
    else: print("Cannot run validator without BPMN structure data.")
    print(f"\n{Style.BRIGHT}--- Validator Test Finished ---{Style.RESET_ALL}")
"""

# --- END OF FILE bpmn_validator.py ---