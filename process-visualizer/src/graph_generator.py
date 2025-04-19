# --- START OF FILE graph_generator.py ---

from os import remove
from os.path import exists
import traceback # Для отладки ошибок

import graphviz
from colorama import Fore # Для цветного вывода

from logging_utils import write_to_file


class GraphGenerator:
    def __init__(self, data, format=None, notebook=False, test_mode=False):

        # Проверка входных данных
        if not isinstance(data, list):
             print(f"{Fore.RED}ERROR: GraphGenerator received invalid data type (expected list, got {type(data)}). Cannot generate graph.{Fore.RESET}")
             # Можно возбудить исключение или установить флаг ошибки
             self.data = []
             self.valid_input = False
             # return # Не можем продолжить инициализацию
        else:
             self.data = data
             self.valid_input = True


        self.bpmn = graphviz.Digraph("bpmn_diagram", filename="bpmn.gv",
                                     graph_attr={'splines': 'ortho'}) # Попробуем ортогональные линии

        if format == "jpeg":
            self.bpmn.format = "jpeg"

        self.notebook = notebook
        self.test_mode = test_mode # Если True, не удаляет старые файлы

        self.last_completed_type = ""      # Тип последнего добавленного элемента (task, exclusive, parallel)
        self.last_completed_node_id = None # ID последнего добавленного узла (T_X, EG_Y_S/E, PG_Z_S/E)

        self.tracker = {} # Для отслеживания связей (для Start/End событий)

        # Глобальные счетчики для уникальных ID
        self.task_counter = 0
        self.exclusive_gateway_counter = 0
        self.parallel_gateway_counter = 0

        # Атрибуты узлов по умолчанию
        self.bpmn.attr(
            "node", shape="box", style="rounded,filled", # Скругленные углы для задач
             color="black", fillcolor="lightgoldenrodyellow" # Другой цвет для задач
        )

    def log_data(self, first_node_id: str, second_node_id: str):
        """Logs connections between node IDs for later analysis (e.g., finding start/end)."""
        if not first_node_id or not second_node_id: return # Не логируем пустые ID

        for node_id in [first_node_id, second_node_id]:
            if node_id not in self.tracker:
                self.tracker[node_id] = {"after": [], "before": []}

        # Избегаем дублирования связей в логгере
        if second_node_id not in self.tracker[first_node_id]["after"]:
            self.tracker[first_node_id]["after"].append(second_node_id)
        if first_node_id not in self.tracker[second_node_id]["before"]:
            self.tracker[second_node_id]["before"].append(first_node_id)

    def connect(self, first_node_id: str, second_node_id: str, label: str | None = None, **kwargs):
        """Creates an edge between two nodes with optional label and logs it."""
        if not first_node_id or not second_node_id:
             print(f"{Fore.YELLOW}Warning: Attempted to connect with invalid node ID(s) ('{first_node_id}' -> '{second_node_id}'). Skipping edge.{Fore.RESET}")
             return

        try:
            # Добавляем стрелку по умолчанию
            kwargs.setdefault('arrowhead', 'normal')
            if label is not None:
                # Добавляем немного отступа к меткам ребер
                self.bpmn.edge(first_node_id, second_node_id, label=f" {label} ", **kwargs)
            else:
                self.bpmn.edge(first_node_id, second_node_id, **kwargs)
            self.log_data(first_node_id, second_node_id)
            # print(f"DEBUG Connect: {first_node_id} -> {second_node_id} (Label: {label})") # Отладка
        except Exception as e:
             print(f"{Fore.RED}Error creating edge: {first_node_id} -> {second_node_id} (Label: {label}). Error: {e}{Fore.RESET}")


    def create_start_and_end_events(self):
        """Adds START and END event nodes based on the connection tracker."""
        if not self.tracker:
             print(f"{Fore.YELLOW}Warning: Connection tracker is empty. Cannot determine START/END events.{Fore.RESET}")
             # Создадим хотя бы один старт и конец для пустого графа? Или вернуть ошибку?
             self.bpmn.node(name="START_0", label="START", shape="circle", style="filled", fillcolor="limegreen", color="black", width="0.5", height="0.5")
             self.bpmn.node(name="END_0", label="END", shape="circle", style="bold,filled", fillcolor="tomato", color="black", width="0.5", height="0.5")
             self.connect("START_0", "END_0")
             return

        start_event_counter = 0
        end_event_counter = 0

        # Атрибуты для событий
        start_attrs = {
            "shape": "circle", "style": "filled", "fillcolor": "limegreen",
            "color": "black", "width": "0.5", "height": "0.5", "fixedsize": "true"
        }
        end_attrs = {
            "shape": "circle", "style": "bold,filled", "fillcolor": "tomato",
            "color": "black", "width": "0.5", "height": "0.5", "fixedsize": "true"
        }

        nodes_with_incoming = set(node for node, data in self.tracker.items() if data.get("before"))
        nodes_with_outgoing = set(node for node, data in self.tracker.items() if data.get("after"))

        # --- Ищем Начальные Узлы ---
        # Узлы без входящих связей (кроме специальных *_E узлов шлюзов)
        start_nodes = set(self.tracker.keys()) - nodes_with_incoming
        # Добавляем задачи T0, если у них нет других входящих связей, кроме возможного цикла
        for node_id, data in self.tracker.items():
            if node_id.startswith("T") and not data.get("before"):
                start_nodes.add(node_id)
            # Особый случай: если T0 имеет входящую связь только от цикла
            elif node_id == "T0" and data.get("before") and all(b.startswith("T") or b.startswith("EG") or b.startswith("PG") for b in data.get("before",[])):
                 is_only_loop = True # Проверить детальнее, если нужно
                 if is_only_loop:
                      start_nodes.add(node_id)


        # Убираем конечные узлы шлюзов из потенциальных стартовых
        start_nodes = {n for n in start_nodes if not (n.endswith("_E"))}

        if not start_nodes and self.tracker: # Если не нашли явного старта, берем первый узел по ID?
             print(f"{Fore.YELLOW}Warning: Could not find explicit start node. Using the node with the lowest ID as start.{Fore.RESET}")
             try:
                  # Попробуем взять T0 или первый шлюз _S
                  potential_starts = sorted([n for n in self.tracker if n=='T0' or (n.endswith('_S') and not self.tracker[n].get('before'))])
                  if potential_starts:
                      start_nodes.add(potential_starts[0])
                  else: # Берем просто первый ключ по сортировке
                      start_nodes.add(sorted(self.tracker.keys())[0])
             except IndexError:
                  print(f"{Fore.RED}Error: Cannot determine start node even from sorted keys.{Fore.RESET}")


        for node_id in sorted(list(start_nodes)): # Сортируем для стабильности
            start_event_id = f"START_{start_event_counter}"
            self.bpmn.node(name=start_event_id, label="", **start_attrs) # Убрал текст "START"
            self.connect(start_event_id, node_id)
            start_event_counter += 1

        # --- Ищем Конечные Узлы ---
        # Узлы без исходящих связей (кроме специальных *_S узлов шлюзов)
        end_nodes = set(self.tracker.keys()) - nodes_with_outgoing
        # Добавляем узлы, у которых исходящие ведут только на START (ошибка в логике?) или на самих себя
        for node_id, data in self.tracker.items():
             outgoing = data.get("after", [])
             if outgoing and all(o.startswith("START_") or o == node_id for o in outgoing):
                  end_nodes.add(node_id)

        # Убираем стартовые узлы шлюзов из потенциальных конечных
        end_nodes = {n for n in end_nodes if not (n.endswith("_S"))}

        # Ищем задачи с добавленным 'process_end_event'
        # (Предполагаем, что ID задачи сохранен в self.tracker)
        # Нужно изменить логику generate_graph, чтобы сохранять ID задач
        # Пока просто ищем узлы без исходящих

        if not end_nodes and self.tracker:
            print(f"{Fore.YELLOW}Warning: Could not find explicit end node. Using node(s) with highest task ID or last gateway ID as end.{Fore.RESET}")
            # Логика определения последнего узла может быть сложной
            # Пока оставим так, Graphviz дорисует как есть

        for node_id in sorted(list(end_nodes)):
            end_event_id = f"END_{end_event_counter}"
            self.bpmn.node(name=end_event_id, label="", **end_attrs) # Убрал текст "END"
            self.connect(node_id, end_event_id)
            end_event_counter += 1

        if start_event_counter == 0 and end_event_counter == 0 and self.tracker:
             print(f"{Fore.YELLOW}Warning: No START or END events were created. The graph might be cyclic or disconnected.{Fore.RESET}")


    def clean_up_graph(self):
        """Removes redundant nodes or edges. (Currently placeholder/simplified)."""
        # Идея: удалить шлюзы с одним входом и одним выходом?
        # Это может быть опасно, если метка на ребре важна.
        # Пока эта функция ничего не делает, чтобы не сломать логику.
        print("Skipping graph clean-up (functionality simplified).")
        # Старый код удалял узлы из файла .gv, что нестабильно.
        # Правильнее модифицировать объект self.bpmn, но это сложнее.
        pass


    # --- Функции contains_nested_lists, dictionary_is_element_of_list, get_nested_lists - вероятно, не нужны ---
    # Логика вложенности теперь обрабатывается в create_bpmn_structure

    # --- Функция dict_is_first_element - вероятно, не нужна ---

    # --- Функция dict_is_direct_child - вероятно, не нужна ---

    # --- Функции count_conditions_in_gateway, check_for_loops_in_gateway, check_for_end_events_in_gateway ---
    # Эти проверки лучше делать на этапе create_bpmn_structure и передавать флаги, если они нужны
    # для специфичной отрисовки шлюзов. Пока уберем их вызовы из handle_gateway.

    def create_node(self, element_id: str, label: str, type: str):
        """Creates a node in the graphviz object and tracks it."""
        if not element_id:
            print(f"{Fore.RED}Error: Attempted to create node with empty ID (Label: '{label}').{Fore.RESET}")
            return

        node_attrs = {}
        # Устанавливаем атрибуты в зависимости от типа узла
        if type == "task":
            node_attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": "lightgoldenrodyellow"}
        elif type == "exclusive_gateway":
            node_attrs = {"shape": "diamond", "style": "filled", "fillcolor": "paleturquoise", "width": "0.6", "height": "0.6", "fixedsize": "true"}
            label = "X" # Стандартное обозначение для XOR-шлюза
        elif type == "parallel_gateway":
            node_attrs = {"shape": "diamond", "style": "filled", "fillcolor": "lightsalmon", "width": "0.6", "height": "0.6", "fixedsize": "true"}
            label = "+" # Стандартное обозначение для AND-шлюза
        else: # Неизвестный тип
             node_attrs = {"shape": "oval", "style": "filled", "fillcolor": "lightgrey"}


        # Устанавливаем атрибуты узла по умолчанию, если они не переопределены
        self.bpmn.attr('node', **node_attrs)
        self.bpmn.node(name=element_id, label=label)
        # print(f"DEBUG Create Node: ID={element_id}, Label='{label}', Type={type}") # Отладка

        # Инициализируем узел в трекере
        if element_id not in self.tracker:
            self.tracker[element_id] = {"after": [], "before": []}


    def handle_task(
        self,
        element: dict,
        parent_gateway_info: dict | None = None, # Передаем информацию о родительском шлюзе
        previous_element_id: str | None = None, # Передаем ID предыдущего узла
        is_first_in_path: bool = False, # Флаг, что это первый элемент в пути шлюза
        is_last_in_path: bool = False # Флаг, что это последний элемент в пути шлюза
    ):
        """Handles a 'task' element from the BPMN structure."""
        content = element.get("content", {})
        agent_info = content.get("agent", {})
        task_info = content.get("task", {})

        # ----- ИЗМЕНЕНИЕ ЗДЕСЬ -----
        # Получаем разрешенное имя агента, используем "Unknown" если что-то не так
        agent = agent_info.get("resolved_word", "Unknown Agent")
        # ----- КОНЕЦ ИЗМЕНЕНИЯ -----

        task_word = task_info.get("word", "Unknown Task")
        task_id = task_info.get("task_id") # Получаем ID задачи (T0, T1, ...)

        if not task_id:
             print(f"{Fore.YELLOW}Warning: Task '{task_word}' is missing 'task_id'. Assigning temporary ID.{Fore.RESET}")
             # Генерируем временный ID, если он отсутствует (не должно происходить)
             task_id = f"TEMP_T{self.task_counter}"
             # Сохраняем ID обратно в элемент для консистентности (если возможно)
             if task_info: task_info['task_id'] = task_id
             self.task_counter += 1 # Увеличиваем счетчик временных ID
        else:
             # Обновляем счетчик задач, если встретили ID больше текущего
             try:
                 num_part = int(task_id[1:])
                 self.task_counter = max(self.task_counter, num_part + 1)
             except (ValueError, IndexError):
                 pass # Игнорируем ID, если он не в формате T<число>

        # Создаем узел задачи
        node_label = f"{agent}: {task_word}"
        self.create_node(element_id=task_id, label=node_label, type="task")

        # --- Логика Соединений ---
        condition = content.get("condition")
        condition_label = condition.get("word") if condition else None

        # 1. Соединение от предыдущего элемента
        if previous_element_id:
            # Не соединяем, если предыдущий был стартовым узлом шлюза (соединение идет оттуда)
            if not previous_element_id.endswith("_S"):
                 self.connect(previous_element_id, task_id)

        # 2. Соединение от родительского шлюза (если это первый элемент пути)
        if parent_gateway_info and is_first_in_path:
            parent_start_node_id = f"{parent_gateway_info['id']}_S"
            # Если есть условие (из эксклюзивного шлюза), используем его как метку
            # Условие должно быть привязано к задаче на этапе create_bpmn_structure
            self.connect(parent_start_node_id, task_id, label=condition_label)

        # 3. Соединение к родительскому шлюзу (если это последний элемент пути)
        #    Не соединяем, если у задачи есть событие завершения процесса
        has_end_event = "process_end_event" in content
        if parent_gateway_info and is_last_in_path and not has_end_event:
             # Не соединяем с конечным узлом родителя, если у родителя есть циклы или конечные события внутри
             parent_has_loops = parent_gateway_info.get("has_loops", False) # Нужны флаги из create_bpmn_structure
             parent_has_end_events = parent_gateway_info.get("has_end_events", False) # Нужны флаги

             # Также проверяем, есть ли у родителя флаг single_condition
             parent_is_single_condition = parent_gateway_info.get("single_condition", False)

             if not parent_has_loops and not parent_has_end_events and not parent_is_single_condition:
                  parent_end_node_id = f"{parent_gateway_info['id']}_E"
                  # Проверяем, существует ли конечный узел (он мог не создаться)
                  if parent_end_node_id in self.tracker:
                       self.connect(task_id, parent_end_node_id)
                  # else: # Конечный узел не создан, задача "зависает" или будет соединена с чем-то следующим

        # Обновляем ID последнего обработанного узла
        self.last_completed_node_id = task_id


    def handle_list(self, children: list, parent_gateway_info: dict):
        """Recursively handles a list of elements within a gateway path."""
        last_processed_id_in_list = None
        num_children = len(children)

        for index, element in enumerate(children):
            is_first = (index == 0)
            is_last = (index == num_children - 1)

            elem_type = element.get("type")

            # Определяем предыдущий элемент *внутри этого списка*
            previous_id_in_this_list = last_processed_id_in_list

            # Добавляем информацию о родительском шлюзе к элементу для передачи дальше
            element['_parent_gateway_info'] = parent_gateway_info

            if elem_type == "task":
                self.handle_task(
                    element=element,
                    parent_gateway_info=parent_gateway_info,
                    previous_element_id=previous_id_in_this_list,
                    is_first_in_path=is_first,
                    is_last_in_path=is_last
                )
                # Обновляем ID последнего элемента в этом списке
                last_processed_id_in_list = element.get("content", {}).get("task", {}).get("task_id")

            elif elem_type == "exclusive" or elem_type == "parallel":
                # Рекурсивный вызов для вложенного шлюза
                self.handle_gateway(
                    element=element,
                    parent_gateway_info=parent_gateway_info,
                    previous_element_id=previous_id_in_this_list,
                    is_first_in_path=is_first,
                    is_last_in_path=is_last
                )
                # Обновляем ID последнего элемента (конечный узел шлюза, если есть)
                end_node_id = f"{element.get('id')}_E"
                last_processed_id_in_list = end_node_id if end_node_id in self.tracker else f"{element.get('id')}_S"

            elif elem_type == "loop":
                content = element.get("content", {})
                go_to_task_id = content.get("go_to")
                condition = content.get("condition")
                condition_label = condition.get("word") if condition else "Loop Back" # Метка для цикла

                if go_to_task_id:
                     # Соединяем от предыдущего элемента к началу цикла (задаче go_to)
                     if previous_id_in_this_list:
                          self.connect(previous_id_in_this_list, go_to_task_id, label=condition_label, style="dashed", constraint="false") # dashed для цикла
                     # Если это первый элемент в пути, соединяем от начала шлюза
                     elif parent_gateway_info:
                           parent_start_node_id = f"{parent_gateway_info['id']}_S"
                           self.connect(parent_start_node_id, go_to_task_id, label=condition_label, style="dashed", constraint="false")
                     else: # Цикл без родителя и предыдущего элемента? Странно.
                           print(f"{Fore.YELLOW}Warning: Loop element points to '{go_to_task_id}' but has no preceding element or parent gateway.{Fore.RESET}")
                else:
                     print(f"{Fore.YELLOW}Warning: Loop element is missing 'go_to' target task ID.{Fore.RESET}")
                # Элемент цикла не создает нового узла и не обновляет last_processed_id_in_list

            elif elem_type == "continue":
                 content = element.get("content", {})
                 go_to_id = content.get("go_to") # ID следующей задачи или шлюза

                 if go_to_id:
                      # Соединяем от начала родительского шлюза к цели 'continue'
                      if parent_gateway_info:
                           parent_start_node_id = f"{parent_gateway_info['id']}_S"
                           # Условие должно быть взято из родительского шлюза по индексу пути
                           path_idx = parent_gateway_info.get('_current_path_idx', -1) # Нужен индекс пути
                           continue_label = parent_gateway_info.get("conditions", [])[path_idx] if 0 <= path_idx < len(parent_gateway_info.get("conditions", [])) else "Continue"
                           self.connect(parent_start_node_id, go_to_id, label=continue_label)
                      else: # 'Continue' без родителя?
                           print(f"{Fore.YELLOW}Warning: 'Continue' element points to '{go_to_id}' but has no parent gateway.{Fore.RESET}")
                 else:
                      print(f"{Fore.YELLOW}Warning: 'Continue' element is missing 'go_to' target ID.{Fore.RESET}")
                 # 'Continue' не создает нового узла и не обновляет last_processed_id_in_list

            else:
                 print(f"{Fore.YELLOW}Warning: Unknown element type '{elem_type}' encountered in list handling.{Fore.RESET}")

            # Удаляем временную информацию о родительском шлюзе
            if '_parent_gateway_info' in element:
                 del element['_parent_gateway_info']


    def handle_gateway(
        self,
        element: dict,
        parent_gateway_info: dict | None = None, # Информация о родительском шлюзе (если есть)
        previous_element_id: str | None = None, # ID предыдущего узла
        is_first_in_path: bool = False, # Флаг, что это первый элемент в пути родителя
        is_last_in_path: bool = False # Флаг, что это последний элемент в пути родителя
    ):
        """Handles a 'parallel' or 'exclusive' gateway element."""
        gateway_id = element.get("id")
        gateway_type = element.get("type") # 'exclusive' или 'parallel'
        children_paths = element.get("children", []) # Список путей (каждый путь - список элементов)

        if not gateway_id or not gateway_type:
             print(f"{Fore.RED}Error: Gateway element is missing 'id' or 'type'. Element: {element}{Fore.RESET}")
             return

        graphviz_type = f"{gateway_type}_gateway"
        start_node_id = f"{gateway_id}_S" # ID начального узла шлюза
        end_node_id = f"{gateway_id}_E"   # ID конечного узла шлюза

        # --- Определяем, нужно ли создавать конечный узел ---
        # Не создаем конечный узел, если:
        # - Внутри есть циклы (обратные связи)
        # - Внутри есть события завершения процесса
        # - Это эксклюзивный шлюз с одним условием (по сути, просто условие на пути)
        has_loops = element.get("has_loops", False) # Нужны флаги из create_bpmn_structure
        has_end_events = element.get("has_end_events", False)
        is_single_condition = element.get("single_condition", False)
        create_end_node = not (has_loops or has_end_events or is_single_condition)
        # --- Конец определения ---

        # Создаем начальный узел шлюза
        # Метка ('X' или '+') устанавливается внутри create_node
        self.create_node(element_id=start_node_id, label="", type=graphviz_type)

        # Создаем конечный узел шлюза (если нужно)
        if create_end_node:
            self.create_node(element_id=end_node_id, label="", type=graphviz_type)
            last_node_for_connection = end_node_id # Для соединения со следующим элементом
        else:
            # Если конечного узла нет, "последним" узлом для соединения будет стартовый
            # или узел, из которого идет цикл/завершение
            last_node_for_connection = start_node_id # По умолчанию

        # --- Соединение от предыдущего элемента ---
        if previous_element_id:
             if not previous_element_id.endswith("_S"): # Не соединяем от начала другого шлюза
                 self.connect(previous_element_id, start_node_id)

        # --- Соединение от родительского шлюза (если это первый элемент пути) ---
        if parent_gateway_info and is_first_in_path:
             parent_start_node_id = f"{parent_gateway_info['id']}_S"
             # Условие перехода к этому шлюзу (если есть)
             condition = element.get("condition")
             condition_label = condition.get("word") if condition else None
             self.connect(parent_start_node_id, start_node_id, label=condition_label)

        # --- Обработка дочерних путей ---
        for path_idx, path_children in enumerate(children_paths):
             # Передаем индекс пути для обработки 'continue' и условий
             element['_current_path_idx'] = path_idx
             # Рекурсивно обрабатываем элементы внутри пути
             self.handle_list(children=path_children, parent_gateway_info=element)
             # Удаляем временный ключ
             if '_current_path_idx' in element: del element['_current_path_idx']


        # --- Соединение с родительским шлюзом (если это последний элемент пути) ---
        # Соединяем только если у текущего шлюза есть конечный узел
        if parent_gateway_info and is_last_in_path and create_end_node:
              # Проверяем флаги родителя
              parent_has_loops = parent_gateway_info.get("has_loops", False)
              parent_has_end_events = parent_gateway_info.get("has_end_events", False)
              parent_is_single_condition = parent_gateway_info.get("single_condition", False)

              if not parent_has_loops and not parent_has_end_events and not parent_is_single_condition:
                   parent_end_node_id = f"{parent_gateway_info['id']}_E"
                   if parent_end_node_id in self.tracker:
                        self.connect(end_node_id, parent_end_node_id)

        # Обновляем ID последнего узла (конечный или начальный шлюз)
        self.last_completed_node_id = last_node_for_connection


    def generate_graph(self):
        """Generates the BPMN graph from the structured data."""
        if not self.valid_input:
             print(f"{Fore.RED}Cannot generate graph due to invalid input data provided during initialization.{Fore.RESET}")
             return

        if not self.data:
             print(f"{Fore.YELLOW}Warning: No BPMN structure data to generate graph from.{Fore.RESET}")
             # Создадим пустой граф со стартом и концом
             self.create_start_and_end_events()
             return


        print("Generating graph nodes and edges...")
        if not self.test_mode:
            self.remove_old_files() # Удаляем старые файлы перед генерацией

        last_processed_node_id = None # ID последнего узла на верхнем уровне

        for global_index, element in enumerate(self.data):
            elem_type = element.get("type")
            previous_element_id = last_processed_node_id

            if elem_type == "task":
                self.handle_task(
                    element=element,
                    previous_element_id=previous_element_id
                    # parent_gateway_info=None, is_first/last=False для верхнего уровня
                )
                # Обновляем ID
                last_processed_node_id = element.get("content", {}).get("task", {}).get("task_id")

            elif elem_type == "parallel" or elem_type == "exclusive":
                self.handle_gateway(
                    element=element,
                    previous_element_id=previous_element_id
                    # parent_gateway_info=None, is_first/last=False для верхнего уровня
                )
                # Обновляем ID (конечный или начальный узел шлюза)
                end_node_id = f"{element.get('id')}_E"
                last_processed_node_id = end_node_id if end_node_id in self.tracker else f"{element.get('id')}_S"
            elif elem_type == "loop": # Циклы не должны быть на верхнем уровне?
                print(f"{Fore.YELLOW}Warning: Loop element found at the top level of BPMN structure. This might indicate an issue.{Fore.RESET}")
                content = element.get("content", {})
                go_to_task_id = content.get("go_to")
                if go_to_task_id and previous_element_id:
                     self.connect(previous_element_id, go_to_task_id, label="Loop Back", style="dashed", constraint="false")
                # Не обновляем last_processed_node_id для цикла
            elif elem_type == "continue": # Continue тоже не должен быть на верхнем уровне
                 print(f"{Fore.YELLOW}Warning: Continue element found at the top level of BPMN structure.{Fore.RESET}")
                 # Не обновляем last_processed_node_id
            else:
                 print(f"{Fore.YELLOW}Warning: Unknown element type '{elem_type}' at top level. Skipping.{Fore.RESET}")


        print("Creating start and end events...")
        self.create_start_and_end_events()
        # self.clean_up_graph() # Пока отключено

        print("Graph generation complete.")
        write_to_file("graph_connection_tracker.json", self.tracker) # Логируем связи


    def remove_old_files(self):
        """Removes previously generated graph files."""
        files_to_remove = [
            "bpmn.gv", "bpmn.gv.pdf", "bpmn.gv.jpeg",
            "cleaned_bpmn.gv", "cleaned_bpmn.gv.pdf", "cleaned_bpmn.gv.jpeg",
             "./src/bpmn.jpeg" # Путь из save_file
        ]
        print("Removing old graph files...")
        for f in files_to_remove:
            if exists(f):
                try:
                    remove(f)
                    # print(f"Removed: {f}")
                except OSError as e:
                    print(f"{Fore.YELLOW}Warning: Could not remove file '{f}'. Error: {e}{Fore.RESET}")


    def show(self):
        """Renders and displays the graph."""
        if not self.valid_input: return
        print("Rendering graph...")
        try:
            # Используем основной файл bpmn.gv, т.к. clean_up_graph отключен
            output_file = self.bpmn.render(filename='bpmn.gv', view=not self.notebook, cleanup=False, quiet=True)
            print(f"Graph saved to {output_file} (and potentially opened).")
        except Exception as e:
             print(f"{Fore.RED}Error rendering or viewing graph: {e}{Fore.RESET}")
             print(f"{Fore.YELLOW}Graphviz engine might not be installed correctly or accessible in PATH.{Fore.RESET}")
             traceback.print_exc()
             # Попробуем просто сохранить .gv файл
             try:
                 self.bpmn.save()
                 print("Saved bpmn.gv file.")
             except Exception as e_save:
                 print(f"{Fore.RED}Failed even to save .gv file: {e_save}{Fore.RESET}")


    def save_file(self, output_path: str = "./src/bpmn.jpeg"):
        """Saves the graph to a specific image file."""
        if not self.valid_input: return
        # Устанавливаем формат перед рендерингом
        self.bpmn.format = output_path.split('.')[-1] if '.' in output_path else 'jpeg'
        print(f"Saving graph to {output_path} (format: {self.bpmn.format})...")
        try:
            # Указываем outfile в render для прямого сохранения
            self.bpmn.render(outfile=output_path, view=False, cleanup=True, quiet=False) # cleanup=True удалит .gv файл
            print(f"Graph successfully saved to {output_path}")
        except Exception as e:
             print(f"{Fore.RED}Error saving graph image to {output_path}: {e}{Fore.RESET}")
             traceback.print_exc()


if __name__ == "__main__":
    # --- Обновленный блок для тестирования ---
    import json
    import os

    structure_file = "output_logs/bpmn_structure/bpmn_final_structure.json" # Используем новое имя
    print(f"--- Running GraphGenerator Test ---")
    print(f"Loading BPMN structure from: {structure_file}")

    if not exists(structure_file):
        print(f"{Fore.RED}ERROR: Input structure file not found: {structure_file}{Fore.RESET}")
    else:
        try:
            with open(structure_file, "r", encoding='utf-8') as file:
                data = json.load(file)

            if not data:
                 print(f"{Fore.YELLOW}Warning: BPMN structure file is empty.{Fore.RESET}")
            else:
                print("BPMN structure loaded successfully.")
                # Создаем экземпляр генератора
                # format='pdf' или 'jpeg', notebook=False для запуска из скрипта
                bpmn_generator = GraphGenerator(data, format='pdf', notebook=False, test_mode=False)

                # Генерируем граф
                bpmn_generator.generate_graph()

                # Отображаем/сохраняем
                bpmn_generator.show() # Откроет PDF (если view=True и есть просмотрщик)
                # bpmn_generator.save_file("./src/test_bpmn_output.jpeg") # Сохранить как JPEG

        except json.JSONDecodeError as e:
            print(f"{Fore.RED}ERROR: Failed to parse JSON from {structure_file}: {e}{Fore.RESET}")
        except Exception as e:
            print(f"{Fore.RED}An unexpected error occurred during GraphGenerator test: {e}{Fore.RESET}")
            traceback.print_exc()

# --- END OF FILE graph_generator.py ---