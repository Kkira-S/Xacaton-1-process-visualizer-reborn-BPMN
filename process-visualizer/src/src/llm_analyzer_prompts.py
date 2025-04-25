# --- START OF FILE llm_analyzer_prompts.py ---

# Системная инструкция для анализа BPMN
# Определяет роль и задачи LLM при анализе
BPMN_ANALYSIS_SYSTEM_MSG = """
You are an expert BPMN validator and process improvement consultant.
You will receive:
1. The original text description of a business process.
2. A structured representation of the BPMN diagram generated from the text (simplified JSON or description).
3. A list of structural errors already found by automated checks (e.g., connectivity, cycles, basic BPMN rules).

Your tasks are:
1. Analyze the structured representation in the context of the original text and the identified structural errors.
2. Identify any *additional* logical inconsistencies, semantic errors, ambiguities, potential deadlocks not caught by structural checks, or areas for process improvement (e.g., redundancies). Pay close attention to whether the diagram accurately reflects the nuances of the original text.
3. For *all* identified errors (both provided and newly found), provide clear, concise, actionable suggestions for fixing them, referencing the original text or BPMN element IDs (like T1, EG0, PG1) where possible.
4. Structure your response clearly using Markdown, separating newly found issues from suggestions for provided errors. If no new issues are found, state that explicitly. If no suggestions can be made for a specific error, indicate that as well.
"""

# Шаблон пользовательского промпта
# Эта функция форматирует входные данные для LLM
def create_bpmn_analysis_prompt(original_text: str, structure_representation: str, structural_errors: list[dict]) -> str:
    """
    Creates the user prompt for the LLM BPMN analysis, combining text, structure, and found errors.

    Args:
        original_text (str): The original text description of the process.
        structure_representation (str): A string representation (e.g., JSON) of the generated BPMN structure.
        structural_errors (list[dict]): A list of dictionaries, where each dictionary represents
                                         an error found by automated checks. Expected keys:
                                         'type', 'element_id', 'message'.

    Returns:
        str: The formatted user prompt string ready for the LLM.
    """

    # Форматируем список ошибок для включения в промпт
    errors_text = "No structural errors found by automated checks."
    if structural_errors:
        errors_list = []
        for i, error in enumerate(structural_errors):
            # Используем .get() для безопасного доступа к ключам
            err_type = error.get('type', 'Unknown Type')
            err_id = error.get('element_id', 'N/A')
            err_msg = error.get('message', 'No details provided.')
            # Формируем строку для каждой ошибки
            errors_list.append(f"{i+1}. Type: `{err_type}`, Element(s): `{err_id}`, Issue: {err_msg}")
        errors_text = "\n".join(errors_list)

    # Собираем финальный промпт с использованием f-строк
    prompt = f"""
Please analyze the following business process and its generated BPMN structure based on the instructions in the system prompt.

**1. Original Process Description:**
```text
    {original_text}
**2. Generated BPMN Structure Representation:**
    {structure_representation}
**3. Structural Errors Found by Automated Checks:**
    {errors_text}
**Analysis Request:**
Based on all the information above, please perform the analysis tasks outlined in the system prompt:
- Identify any additional logical/semantic issues.
- Provide specific suggestions to fix *all* errors (provided and new).

**Format your response using markdown:**
### Newly Identified Issues:
- [List any new logical/semantic problems found]

### Suggestions for Correction:
- **Error [Error Index or ID from above / New Issue ID]:** [Your suggestion for fixing this specific error]
- **Error [...]:** [Suggestion]
...
"""
    return prompt

# --- END OF FILE llm_analyzer_prompts.py ---