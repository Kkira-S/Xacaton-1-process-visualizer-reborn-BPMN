# --- START OF FILE graph_generator.py ---
import os
import sys
from os import remove, makedirs
from os.path import exists, dirname, join
import traceback
from typing import List, Dict, Optional, Set, Tuple, Any
import graphviz
from colorama import Fore, Style, init
import re
import networkx as nx  # Убедимся, что импортирован

# Инициализация colorama
init(autoreset=True)

# Импорт утилиты логирования (если она нужна)
try:
    from logging_utils import write_to_file

    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False


    def write_to_file(filename, data):
        pass  # Заглушка


class GraphGenerator:
    """
    Генерирует визуализацию BPMN-подобной диаграммы из структурированных данных
    с использованием Graphviz. Версия с polyline и без constraint=false для loop.
    """

    def __init__(self, data: List[Dict], format: str = 'svg', output_dir: str = "output_graphs",
                 filename: str = "bpmn_diagram"):
        if not isinstance(data, list):
            print(f"{Fore.RED}ERROR: Invalid input data type.{Style.RESET_ALL}")
            self.data = [];
            self.valid_input = False
        else:
            self.data = data; self.valid_input = True

        self.output_dir = output_dir;
        self.filename = filename
        self.gv_filepath = os.path.abspath(join(self.output_dir, f"{self.filename}.gv"))

        self.bpmn = graphviz.Digraph(
            name="bpmn_diagram", filename=self.gv_filepath,
            graph_attr={
                'rankdir': 'TB',  # Сверху вниз
                # --- ИЗМЕНЕНИЕ: Используем polyline ---
                'splines': 'polyline',
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                # --- ИЗМЕНЕНИЕ: Увеличены отступы ---
                'nodesep': '0.8',  # Расстояние между узлами
                'ranksep': '1.0',  # Расстояние между рангами
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                'bgcolor': 'transparent',
                'pad': '0.5',
                'compound': 'true',  # Для кластеров
                'newrank': 'true',
                'fontname': 'Arial',
                'fontsize': '10'
            },
            node_attr={'fontname': 'Arial', 'fontsize': '10', 'shape': 'box', 'style': 'rounded,filled',
                       'color': '#333333', 'fillcolor': '#FFFFE0', 'height': '0.5', 'width': '1.2',
                       'margin': '0.15,0.05'},
            edge_attr={'fontname': 'Arial', 'fontsize': '9', 'color': '#555555', 'arrowhead': 'normal',
                       'arrowsize': '0.7'}
        )
        self.bpmn.format = format.lower()

        self.created_nodes: Set[str] = set();
        self.edges_log: Set[Tuple[str, str]] = set()
        self.lanes_info: Dict[str, List[str]] = {};
        self.subgraphs: Dict[str, Any] = {}
        self.node_agent_map: Dict[str, str] = {}
        self.last_node_in_path: Dict[str, Optional[str]] = {}
        self.gateway_stack: List[Dict] = []
        self.gateway_nodes: Dict[str, Dict[str, str]] = {}
        self.element_map_by_id: Dict[str, Dict] = {}
        self._build_element_map(self.data)
        self.graph: Optional[nx.DiGraph] = None

    def _build_element_map(self, elements: List[Dict]):
        if not isinstance(elements, list): return
        for element in elements:
            if not isinstance(element, dict): continue
            elem_id = None;
            graph_id = None;
            elem_type = element.get("type")
            if elem_type == "task":
                elem_id = element.get("content", {}).get("task", {}).get("task_id"); graph_id = elem_id
            elif elem_type in ["exclusive", "parallel"]:
                elem_id = element.get("id");
                if elem_id: self.element_map_by_id[f"{elem_id}_S"] = element; self.element_map_by_id[
                    f"{elem_id}_E"] = element; graph_id = elem_id
            if graph_id: self.element_map_by_id[graph_id] = element
            if "children" in element and isinstance(element.get("children"), list):
                for child_path in element["children"]: self._build_element_map(child_path)

    def _clean_filename(self, name: str) -> str:
        name = re.sub(r'\W+', '_', name);
        name = name.strip('_');
        if name and name[0].isdigit(): name = '_' + name
        return name if name else "unknown"

    def _get_agent_subgraph(self, agent_name: str) -> Any:
        if not agent_name or agent_name == "Unknown Agent": return self.bpmn
        safe_agent_name = self._clean_filename(agent_name);
        cluster_name = f"cluster_{safe_agent_name}"
        if cluster_name not in self.subgraphs:
            subgraph = graphviz.Digraph(name=cluster_name,
                                        graph_attr={'label': agent_name, 'style': 'filled', 'color': 'lightgrey',
                                                    'fontsize': '12', 'fontcolor': '#333333', 'margin': '15'})
            self.subgraphs[cluster_name] = subgraph;
            self.lanes_info[agent_name] = []
            return subgraph
        else:
            return self.subgraphs[cluster_name]

    def _add_node(self, node_id: str, label: str, agent: Optional[str] = None, **attrs):
        if not node_id or node_id in self.created_nodes: return
        target_graph = self._get_agent_subgraph(agent) if agent else self.bpmn
        if agent and agent != "Unknown Agent": self.lanes_info.setdefault(agent, []).append(node_id);
        self.node_agent_map[node_id] = agent

        try:
            max_len = 30;
            words = label.split();
            display_label = "";
            current_line = ""
            for word in words:
                if not current_line:
                    current_line = word
                elif len(current_line) + len(word) + 1 <= max_len:
                    current_line += " " + word
                else:
                    display_label += current_line + "\n"; current_line = word
            display_label += current_line
            display_label = re.sub(r"^[^:]+:\s*", "", display_label).strip()

            node_type = attrs.get('data_type', 'task');
            specific_attrs = {}
            if node_type == "task":
                specific_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFFE0',
                                  'label': display_label}
            elif node_type == "exclusive_gateway":
                specific_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#add8e6', 'width': '0.6',
                                  'height': '0.6', 'fixedsize': 'true', 'label': 'X'}
            elif node_type == "parallel_gateway":
                specific_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#ffdab9', 'width': '0.6',
                                  'height': '0.6', 'fixedsize': 'true', 'label': '+'}
            elif node_type == "start_event":
                specific_attrs = {'shape': 'circle', 'style': 'filled', 'fillcolor': '#90ee90', 'label': '',
                                  'width': '0.4', 'height': '0.4', 'fixedsize': 'true'}
            elif node_type == "end_event":
                specific_attrs = {'shape': 'doublecircle', 'style': 'bold,filled', 'fillcolor': '#ff6347', 'label': '',
                                  'width': '0.4', 'height': '0.4', 'fixedsize': 'true'}
            elif node_type == "implicit":
                specific_attrs = {'shape': 'plaintext', 'label': display_label}
            else:
                specific_attrs = {'label': display_label}

            final_attrs = {**specific_attrs, **attrs};
            if 'data_type' in final_attrs: del final_attrs['data_type']
            if 'label' in attrs and node_type not in ['task', 'implicit']:
                if 'label' in final_attrs: del final_attrs['label']

            target_graph.node(name=node_id, **final_attrs)
            self.created_nodes.add(node_id)
        except Exception as e:
            print(f"{Fore.RED}Error adding node '{node_id}': {e}{Style.RESET_ALL}")

    def _add_edge(self, u: str, v: str, label: Optional[str] = None, **attrs):
        if not u or not v or (u, v) in self.edges_log: return
        if u not in self.created_nodes: self._add_node_placeholder(u)
        if v not in self.created_nodes: self._add_node_placeholder(v)

        try:
            edge_attrs = {'arrowhead': 'normal'};
            data_type = attrs.get('data_type')
            if data_type == 'loop' or data_type == 'continue':
                # --- ИЗМЕНЕНИЕ: Убираем constraint=false ---
                edge_attrs.update({'style': 'dashed', 'arrowhead': 'empty'})
                # edge_attrs.update({'style': 'dashed', 'constraint': 'false', 'arrowhead': 'empty'})
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            edge_attrs.update({k: v for k, v in attrs.items() if k != 'data_type'})
            display_label = label.strip() if label and label.strip() else None
            xlabel_attr = {'xlabel': f" {display_label} "} if display_label else {}
            self.bpmn.edge(u, v, **edge_attrs, **xlabel_attr)
            self.edges_log.add((u, v))
        except Exception as e:
            print(f"{Fore.RED}Error creating edge: {u} -> {v} (Label: {label}). Error: {e}{Style.RESET_ALL}")

    def _add_node_placeholder(self, node_id: str):
        if node_id in self.created_nodes: return
        label = f"Implicit({node_id})";
        data_type = "implicit";
        agent = None
        element_data = self.element_map_by_id.get(node_id)
        if not element_data and (
                node_id.endswith("_S") or node_id.endswith("_E")): element_data = self.element_map_by_id.get(
            node_id[:-2])
        if node_id.startswith("T"):
            data_type = "task"; label = node_id
        elif node_id.endswith("_S"):
            data_type = "exclusive_gateway" if node_id.startswith("EG") else "parallel_gateway"; label = ""
        elif node_id.endswith("_E"):
            data_type = "exclusive_gateway" if node_id.startswith("EG") else "parallel_gateway"; label = ""
        if element_data and data_type == "task":
            agent = self._get_element_agent_from_data(element_data); label = element_data.get("content", {}).get("task",
                                                                                                                 {}).get(
                "word", label)
        elif element_data and data_type != "implicit":
            agent = self._get_element_agent(node_id)
        self._add_node(node_id, label=label, agent=agent, data_type=data_type)

    def _get_element_agent_from_data(self, element_data: Dict) -> Optional[str]:
        agent = element_data.get('content', {}).get('agent', {}).get('resolved_word')
        return agent if agent and agent != "Unknown Agent" else None

    def _get_element_agent(self, element_id: str) -> Optional[str]:
        element_data = self.element_map_by_id.get(element_id);
        resolved_agent = None
        if element_data and element_data.get("type") == 'task': return self._get_element_agent_from_data(element_data)
        if (element_id.endswith("_S") or element_id.endswith("_E")) and self.graph:
            if element_id.endswith("_S") and self.graph.has_node(element_id):
                try:
                    for succ_id in self.graph.successors(element_id):
                        succ_node_data = self.graph.nodes.get(succ_id, {});
                        if succ_node_data.get('type') == 'task':
                            task_element_data = succ_node_data.get('element_data')
                            if task_element_data: agent = self._get_element_agent_from_data(task_element_data)
                            if agent: return agent
                except:
                    pass
            base_id = element_id[:-2];
            element_data_base = self.element_map_by_id.get(base_id)
            if element_data_base and element_id.endswith("_S") and self.graph and self.graph.has_node(element_id):
                try:
                    for succ_id in self.graph.successors(element_id):
                        succ_node_data = self.graph.nodes.get(succ_id, {})
                        if succ_node_data.get('type') == 'task':
                            task_element_data = succ_node_data.get('element_data')
                            if task_element_data: agent = self._get_element_agent_from_data(task_element_data)
                            if agent: return agent
                except:
                    pass
        return None

    def _process_recursive(self, elements: List[Dict], last_node_id: Optional[str]) -> Optional[str]:
        current_last_node = last_node_id
        for element in elements:
            if not isinstance(element, dict): continue
            elem_type = element.get("type");
            new_last_node = None
            if elem_type == "task":
                content = element.get("content", {});
                task_info = content.get("task", {});
                task_id = task_info.get("task_id")
                if not task_id: continue
                agent = self._get_element_agent_from_data(element);
                label = task_info.get("word", "Unnamed Task")
                self._add_node(task_id, label=label, agent=agent, data_type="task")
                if current_last_node:
                    condition_label = None
                    if current_last_node.endswith("_S") and self.gateway_stack:
                        parent_gateway = self.gateway_stack[-1];
                        parent_id = parent_gateway.get("id")
                        if parent_id and current_last_node == self.gateway_nodes.get(parent_id, {}).get('start'):
                            path_idx = parent_gateway.get('_current_path_idx', -1);
                            conditions = parent_gateway.get("conditions", [])
                            if parent_gateway.get("type") == "exclusive" and 0 <= path_idx < len(
                                conditions): condition_label = conditions[path_idx]
                    self._add_edge(current_last_node, task_id, label=condition_label)
                new_last_node = task_id
            elif elem_type in ["exclusive", "parallel"]:
                gateway_id = element.get("id");
                if not gateway_id: continue
                gw_type = "exclusive_gateway" if elem_type == "exclusive" else "parallel_gateway";
                start_node_id = f"{gateway_id}_S";
                end_node_id = f"{gateway_id}_E"
                self.gateway_nodes[gateway_id] = {'start': start_node_id, 'end': end_node_id}
                gateway_agent = self._get_element_agent(start_node_id)
                self._add_node(start_node_id, label="", agent=gateway_agent, data_type=gw_type)
                self._add_node(end_node_id, label="", agent=gateway_agent, data_type=gw_type)
                if current_last_node:
                    condition_label = None
                    if current_last_node.endswith("_S") and self.gateway_stack:
                        parent_gateway = self.gateway_stack[-1];
                        parent_id = parent_gateway.get("id")
                        if parent_id and current_last_node == self.gateway_nodes.get(parent_id, {}).get('start'):
                            path_idx = parent_gateway.get('_current_path_idx', -1);
                            conditions = parent_gateway.get("conditions", [])
                            if parent_gateway.get("type") == "exclusive" and 0 <= path_idx < len(
                                conditions): condition_label = conditions[path_idx]
                    self._add_edge(current_last_node, start_node_id, label=condition_label)

                self.gateway_stack.append(element);
                children_paths = element.get("children", [])
                if not children_paths:
                    self._add_edge(start_node_id, end_node_id)
                else:
                    conditions = element.get("conditions", [])
                    path_end_nodes = []
                    for idx, path in enumerate(children_paths):
                        element['_current_path_idx'] = idx
                        # --- ИЗМЕНЕНИЕ: Передаем start_node_id как начало пути ---
                        last_node_in_path = self._process_recursive(path, start_node_id)
                        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                        path_end_nodes.append(last_node_in_path)

                        path_condition_label = conditions[idx] if elem_type == "exclusive" and idx < len(
                            conditions) else None
                        # --- ИЗМЕНЕНИЕ: Ребро от _S к первому элементу пути добавляется внутри рекурсивного вызова ---
                        # (см. логику добавления ребра в начале обработки task и gateway)
                        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                        # Соединяем последний узел пути с конечным узлом шлюза (_E)
                        if last_node_in_path and last_node_in_path != start_node_id:
                            self._add_edge(last_node_in_path, end_node_id)
                        # Если путь пустой или прерван, соединяем S и E с условием (если есть)
                        elif not last_node_in_path or last_node_in_path == start_node_id:
                            self._add_edge(start_node_id, end_node_id, label=path_condition_label)

                if '_current_path_idx' in element: del element['_current_path_idx']
                self.gateway_stack.pop();
                new_last_node = end_node_id
            elif elem_type == "loop":
                go_to_task_id = element.get("content", {}).get("go_to");
                if go_to_task_id and current_last_node: self._add_edge(current_last_node, go_to_task_id, label="Loop",
                                                                       data_type='loop')
                new_last_node = None  # Прерываем поток
            elif elem_type == "continue":
                go_to_id = element.get("content", {}).get("go_to");
                target_node_id = None
                if go_to_id:
                    if go_to_id.startswith("T"):
                        target_node_id = go_to_id
                    elif go_to_id.startswith(("EG", "PG")):
                        target_node_id = f"{go_to_id}_S"
                parent_gateway_start_node = None;
                condition_label = "Continue"
                if self.gateway_stack:
                    parent_gateway = self.gateway_stack[-1];
                    parent_id = parent_gateway.get('id')
                    if parent_id: parent_gateway_start_node = self.gateway_nodes.get(parent_id, {}).get('start')
                    path_idx = parent_gateway.get('_current_path_idx', -1);
                    conditions = parent_gateway.get("conditions", [])
                    if parent_gateway.get("type") == "exclusive" and 0 <= path_idx < len(conditions): condition_label = \
                    conditions[path_idx]
                if parent_gateway_start_node and target_node_id: self._add_edge(parent_gateway_start_node,
                                                                                target_node_id, label=condition_label,
                                                                                data_type='continue')
                new_last_node = None  # Прерываем поток

            if new_last_node is not None:
                current_last_node = new_last_node
            elif elem_type in ['loop', 'continue']:
                current_last_node = None
        return current_last_node

    def generate_graph(self, graph_from_validator: Optional[nx.DiGraph] = None):
        """Генерирует граф Graphviz."""
        if not self.valid_input or not self.data:
            print(f"{Fore.YELLOW}Warning: No valid data to generate graph.{Style.RESET_ALL}")
            self._add_node("START_EVENT", "", data_type="start_event");
            self._add_node("END_EVENT", "", data_type="end_event");
            self._add_edge("START_EVENT", "END_EVENT")
            return
        self.graph = graph_from_validator
        print("Generating Graphviz structure...")
        if not exists(self.output_dir):
            try:
                makedirs(self.output_dir)
            except OSError as e:
                print(f"{Fore.RED}Error creating output dir: {e}{Style.RESET_ALL}")

        self._add_node("START_EVENT", "", data_type="start_event")
        self._add_node("END_EVENT", "", data_type="end_event")
        # Запускаем рекурсивную обработку от Start события
        final_node = self._process_recursive(self.data, "START_EVENT")
        # Соединяем последний узел основного потока с End
        if final_node and final_node != "START_EVENT":
            is_open_gateway_start = False
            if final_node.endswith("_S"):
                gw_id = final_node[:-2]
                if gw_id in self.gateway_nodes and self.gateway_nodes[gw_id][
                    'end'] not in self.created_nodes: is_open_gateway_start = True
            if not is_open_gateway_start:
                self._add_edge(final_node, "END_EVENT")
        elif not final_node and len(self.data) > 0:
            print(f"{Fore.YELLOW}Warning: Main flow broken.{Style.RESET_ALL}")
        elif final_node == "START_EVENT":
            self._add_edge("START_EVENT", "END_EVENT")

        # Добавляем сабграфы (дорожки)
        for subgraph in self.subgraphs.values(): self.bpmn.subgraph(subgraph)
        print("Graphviz structure generated.")

    def render_graph(self, view=False) -> Optional[str]:
        """Рендерит граф и возвращает путь к файлу."""
        if not self.valid_input: return None
        output_path = f"{self.gv_filepath}.{self.bpmn.format}"
        output_file_dir = dirname(output_path)
        if not exists(output_file_dir):
            try:
                makedirs(output_file_dir); print(f"Created directory: {output_file_dir}")
            except OSError as e:
                print(f"{Fore.RED}Error creating dir: {e}{Style.RESET_ALL}"); return None
        print(f"Rendering graph to {output_path}...")
        try:
            rendered_path = self.bpmn.render(outfile=output_path, view=view, cleanup=True, quiet=False)
            print(f"{Fore.GREEN}Graph rendered to: {rendered_path}{Style.RESET_ALL}")
            return rendered_path
        except graphviz.backend.execute.ExecutableNotFound:
            print(f"{Fore.RED}ERROR: Graphviz not found.{Style.RESET_ALL}");
            print(f"{Fore.YELLOW}Install Graphviz (https://graphviz.org/download/) and add to PATH.{Style.RESET_ALL}")
            try:
                self.bpmn.save(); print(f"Saved Graphviz source: {self.gv_filepath}")
            except Exception as e_save:
                print(f"{Fore.RED}Failed to save .gv: {e_save}{Style.RESET_ALL}")
            return None
        except Exception as e:
            print(f"{Fore.RED}Error rendering graph: {e}{Style.RESET_ALL}");
            traceback.print_exc()
            try:
                self.bpmn.save(); print(f"Saved Graphviz source: {self.gv_filepath}")
            except:
                pass
            return None


# --- Тестовый блок ---
if __name__ == "__main__":
    import json
    print(f"{Style.BRIGHT}--- Running GraphGenerator Enhanced Test with YOUR data ---{Style.RESET_ALL}")
    current_dir = dirname(os.path.abspath(__file__))

    # --- ИЗМЕНЕНИЕ: Используем ваши данные напрямую ---
    your_test_data = [
        { "content": {"agent": {"original_word": "The customer", "resolved_word": "The customer", "entity": {"entity_group": "AGENT", "score": 0.9952741861343384, "word": "The customer", "start": 0, "end": 12}}, "task": {"entity_group": "TASK", "score": 0.9951665997505188, "word": "decides if he wants to finance or pay in cash", "start": 13, "end": 58, "task_id": "T0"}, "sentence_idx": 0}, "type": "task" },
        { "content": {"agent": {"original_word": "the customer", "resolved_word": "the customer", "entity": {"entity_group": "AGENT", "score": 0.9955123662948608, "word": "the customer", "start": 96, "end": 108}}, "task": {"entity_group": "TASK", "score": 0.9956878423690796, "word": "fill out a loan application", "start": 122, "end": 149, "task_id": "T1"}, "sentence_idx": 1, "condition": {"entity_group": "CONDITION", "score": 0.9929850101470947, "word": "If the customer chooses to finance", "start": 60, "end": 94, "condition_id": "C0"}}, "type": "task" },
        { "content": {"agent": {"original_word": "the customer", "resolved_word": "the customer", "entity": {"entity_group": "AGENT", "score": 0.9960280656814575, "word": "the customer", "start": 163, "end": 175}}, "task": {"entity_group": "TASK", "score": 0.9967973232269287, "word": "sends the application", "start": 176, "end": 197, "task_id": "T2"}, "sentence_idx": 2}, "type": "task" },
        { "content": {"agent": {"original_word": "the customer", "resolved_word": "the customer", "entity": {"entity_group": "AGENT", "score": 0.9956063628196716, "word": "the customer", "start": 251, "end": 263}}, "task": {"entity_group": "TASK", "score": 0.9377094507217407, "word": "bring the total cost of the car", "start": 277, "end": 308, "task_id": "T3"}, "sentence_idx": 3, "condition": {"entity_group": "CONDITION", "score": 0.9931746125221252, "word": "If the customer chooses to pay in cash", "start": 211, "end": 249, "condition_id": "C1"}}, "type": "task" },
        { "content": {"agent": {"original_word": "the customer", "resolved_word": "the customer", "entity": {"entity_group": "AGENT", "score": 0.9963616728782654, "word": "the customer", "start": 422, "end": 434}}, "task": {"entity_group": "TASK", "score": 0.9961073398590088, "word": "sign the contract", "start": 440, "end": 457, "task_id": "T4"}, "sentence_idx": 4}, "type": "task" }
    ]
    print("Using YOUR provided test data.")
    data_to_use = your_test_data
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # --- ГЕНЕРАЦИЯ ГРАФА (Опционально, для _get_element_agent) ---
    # Для ваших данных граф не обязателен, т.к. нет шлюзов для определения агента
    graph_for_agents = None
    print("Skipping graph building from validator for this test.")
    # --- КОНЕЦ ГЕНЕРАЦИИ ГРАФА ---


    # --- Генерация Визуализации ---
    try:
        print("\nGenerating Graphviz visualization...")
        output_directory = join(current_dir, "..", "output_graphs")
        # --- ИЗМЕНЕНИЕ: Имя файла для ваших данных ---
        output_filename = "your_data_visualization"
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        bpmn_generator = GraphGenerator(data_to_use, format='svg', output_dir=output_directory, filename=output_filename)
        # Передаем граф валидатора, если он есть (в данном случае None)
        bpmn_generator.generate_graph(graph_from_validator=graph_for_agents)
        rendered_file = bpmn_generator.render_graph(view=True) # Пытаемся открыть
        if rendered_file: print(f"Check the generated file: {rendered_file}")

    except Exception as e: print(f"{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}"); traceback.print_exc()
    print(f"{Style.BRIGHT}--- GraphGenerator Test Finished ---{Style.RESET_ALL}")
