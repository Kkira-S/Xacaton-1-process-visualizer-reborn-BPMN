# --- START OF FILE main.py ---

import argparse
import json
import os
import sys
import traceback

# Импорт необходимых функций и классов из ваших файлов
try:
    from process_bpmn_data import process_text
except ImportError:
    print("ERROR: Could not import 'process_text' from 'process_bpmn_data.py'. Make sure the file exists and is in the Python path.")
    sys.exit(1)

try:
    # Импортируем сам класс и флаг доступности LLM
    from bpmn_validator import BPMNValidator, LLM_AVAILABLE
except ImportError:
    print("ERROR: Could not import 'BPMNValidator' from 'bpmn_validator.py'. Make sure the file exists and is in the Python path.")
    # Устанавливаем заглушки, чтобы код ниже не падал сразу, но валидация не будет работать
    BPMNValidator = None
    LLM_AVAILABLE = False
    print("WARNING: BPMN validation will be skipped.")

# Импорт colorama для цветного вывода (опционально, но полезно)
try:
    from colorama import Fore, Style, init
    init(autoreset=True) # Инициализация colorama
except ImportError:
    # Заглушки, если colorama не установлена
    print("Warning: 'colorama' not found. Output will not be colored.")
    class Fore: Style=Fore; RED=YELLOW=GREEN=CYAN=MAGENTA=BLUE=BRIGHT="" # Добавлен BRIGHT
    class Style: RESET_ALL=BRIGHT="" # Добавлен BRIGHT

def parse_arguments():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Process textual BPMN descriptions, validate, and optionally save the result.")

    # Аргументы для ввода текста
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--text", help="Textual description of a process")
    input_group.add_argument("-f", "--file", help="Path to a file containing a textual description of a process")

    # Аргументы для управления валидатором
    parser.add_argument(
        "--skip-validation",
        help="Skip the validation step entirely.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-llm",
        help="Skip LLM analysis during validation (if LLM is available).",
        action="store_true",
    )
    parser.add_argument(
        "--skip-fixes",
        help="Skip applying automatic safe fixes during validation.",
        action="store_true",
    )

    # Аргумент для сохранения результата
    parser.add_argument(
        "-o",
        "--output-file",
        help="Path to save the final (validated/corrected) BPMN structure as JSON. "
             "If not provided, the structure is only processed.",
        default=None # По умолчанию не сохраняем
    )

    args = parser.parse_args()

    # Чтение текста из файла, если указан
    text_input = ""
    if args.text:
        text_input = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding='utf-8') as f:
                text_input = f.read()
            print(f"{Fore.GREEN}Read process description from: {args.file}{Style.RESET_ALL}")
        except FileNotFoundError:
            print(f"{Fore.RED}Error: Input file not found at '{args.file}'{Style.RESET_ALL}")
            sys.exit(1) # Выход, если файл не найден
        except Exception as e:
            print(f"{Fore.RED}Error reading file '{args.file}': {e}{Style.RESET_ALL}")
            traceback.print_exc()
            sys.exit(1)

    if not text_input:
         print(f"{Fore.RED}Error: No input text provided or read from file.{Style.RESET_ALL}")
         sys.exit(1)

    return text_input, args

if __name__ == "__main__":
    print(f"{Style.BRIGHT}--- Starting BPMN Processing and Validation ---{Style.RESET_ALL}")

    original_text, args = parse_arguments()
    bpmn_structure = None
    final_structure = None

    # --- Шаг 1: Обработка текста для получения начальной структуры BPMN ---
    try:
        print("\n---> Step 1: Processing text to generate initial BPMN structure...")
        bpmn_structure = process_text(original_text) # Эта функция печатает свои логи

        if bpmn_structure is None:
            print(f"{Fore.RED}Error: Failed to generate initial BPMN structure from text. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        else:
            print(f"{Fore.GREEN}Successfully generated initial BPMN structure.{Style.RESET_ALL}")

        final_structure = bpmn_structure # По умолчанию, финальная = начальная

    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}An critical error occurred during initial text processing:{Style.RESET_ALL}")
        print(f"{Fore.RED}{e}{Style.RESET_ALL}")
        traceback.print_exc()
        sys.exit(1)


    # --- Шаг 2: Валидация и возможное исправление структуры ---
    if not args.skip_validation and BPMNValidator:
        print("\n---> Step 2: Validating the generated BPMN structure...")
        try:
            validator = BPMNValidator(bpmn_structure, original_text)

            # Определяем, запускать ли LLM и исправления
            run_llm_validation = not args.skip_llm and LLM_AVAILABLE
            run_auto_fixes = not args.skip_fixes

            if not LLM_AVAILABLE and not args.skip_llm:
                print(f"{Fore.YELLOW}LLM analysis requested, but LLM is not available/loaded. Skipping LLM part of validation.{Style.RESET_ALL}")

            print(f"Running validation with: LLM Analysis={run_llm_validation}, Auto Fixes={run_auto_fixes}")

            # --- ИЗМЕНЕНИЕ: Получаем все три списка ---
            errors, warnings, suggestions = validator.validate(run_llm=run_llm_validation, run_fixes=run_auto_fixes)

            # Получаем финальную (возможно исправленную) структуру
            final_structure = validator.get_corrected_structure()

            # --- ИЗМЕНЕНИЕ: Улучшенный вывод итогов валидации ---
            print("\nValidation Summary:")
            if errors:
                print(f"  {Fore.RED}Found {len(errors)} structural error(s).{Style.RESET_ALL}")
            else:
                print(f"  {Fore.GREEN}No structural errors found.{Style.RESET_ALL}")

            if warnings:
                print(f"  {Fore.YELLOW}Found {len(warnings)} warning(s).{Style.RESET_ALL}")
            else:
                print(f"  No warnings found.")

            if suggestions:
                 print(f"  {Fore.CYAN}LLM provided {len(suggestions)} suggestion(s).{Style.RESET_ALL}")
            elif run_llm_validation: # Если LLM запускался, но предложений нет
                 print(f"  LLM analysis ran, but provided no suggestions.")
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}An error occurred during BPMN validation:{Style.RESET_ALL}")
            print(f"{Fore.RED}{e}{Style.RESET_ALL}")
            traceback.print_exc()
            # Продолжаем выполнение, но финальная структура может быть некорректной
            print(f"{Fore.YELLOW}Validation failed. Proceeding with the structure obtained before validation.{Style.RESET_ALL}")
            final_structure = bpmn_structure # Возвращаемся к структуре до валидации
    elif args.skip_validation:
        print("\n---> Step 2: Validation skipped by user request.")
    else: # BPMNValidator не был импортирован
         print("\n---> Step 2: Validation skipped because BPMNValidator could not be loaded.")


    # --- Шаг 3: Сохранение финальной структуры (если запрошено) ---
    if args.output_file and final_structure:
        print(f"\n---> Step 3: Saving final BPMN structure to {args.output_file}...")
        output_dir = os.path.dirname(args.output_file)
        try:
            # Создаем директорию, если ее нет
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            with open(args.output_file, "w", encoding='utf-8') as f:
                json.dump(final_structure, f, indent=2, ensure_ascii=False)
            print(f"{Fore.GREEN}Final structure successfully saved.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving final structure to '{args.output_file}': {e}{Style.RESET_ALL}")
            traceback.print_exc()
    elif not final_structure:
         print(f"\n{Fore.YELLOW}Warning: Cannot save output file because the final structure is missing.{Style.RESET_ALL}")


    # --- Шаг 4: Генерация Графики (если нужно) ---
    # generate_graph_pdf(output, notebook) - удален по запросу
    # Пример вызова generate_graph_image (раскомментировать и настроить при необходимости)
    # generate_image = False # Управлять через аргумент командной строки?
    # if generate_image and final_structure:
    #     try:
    #         from process_bpmn_data import generate_graph_image
    #         print("\n---> Optional Step: Generating graph image...")
    #         generate_graph_image(final_structure) # Эта функция сохранит файл сама
    #     except ImportError:
    #         print(f"{Fore.YELLOW}Warning: 'generate_graph_image' not found. Skipping image generation.{Style.RESET_ALL}")
    #     except Exception as e:
    #         print(f"{Fore.RED}Error generating graph image: {e}{Style.RESET_ALL}")
    #         traceback.print_exc()

    print(f"\n{Style.BRIGHT}--- Processing Finished ---{Style.RESET_ALL}")

# --- END OF FILE main.py ---