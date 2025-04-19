import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True, torch_dtype=torch.float16)
model.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

SYSTEM_MSG = ("You are a highly experienced business process modelling expert, specializing in BPMN modelling. "
              "You will be provided with a description of a complex business process and will need to answer questions regarding the process."
              " Your answers should be clear, accurate and concise.")

def extract_exclusive_gateways(process_description: str) -> str:
    """
    Извлекает текст задач, следующих из условий указанного exclusive gateway,
    включая не-непосредственные задачи.
    """
    system_msg = (
        "You are a highly experienced business process modelling expert, "
        "specializing in BPMN modelling. You will be provided with a description of a complex business process "
        "and will need to extract the text which belongs to a specific exclusive gateway. "
        "You have to extract all the tasks that follow from the conditions in the exclusive gateway, not only the immediate tasks."
    )

    user_msg = (
        "Process: 'If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. "
        "If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase.'\n"
        "Exclusive gateway 1: If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. "
        "If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase.\n\n"
        "Process: 'If the customer chooses to finance, the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction. "
        "After the financial decision has been made, if the customer decides to trade in their old car, the dealership will provide an appraisal and deduct the value from the total cost of the new car. However, if the customer chooses not to trade in their old car, they will need to pay the full price of the new car.'\n"
        "Exclusive gateway 1: If the customer chooses to finance, the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction.\n"
        "Exclusive gateway 2: if the customer decides to trade in their old car, the dealership will provide an appraisal and deduct the value from the total cost of the new car. However, if the customer chooses not to trade in their old car, they will need to pay the full price of the new car.\n\n"
        "Process: 'If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.'\n"
        "Exclusive gateway 1: If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.\n\n"
        "Process: 'If the company chooses to create a new product, the company designs the product. If the company is satisfied with the design, the company launches the product and the process ends. "
        "If not, the company redesigns the product. On the other hand, if the company chooses to modify an existing product, the company chooses a product to redesign and then redesigns the product.'\n"
        "Exclusive gateway 1: If the company chooses to create a new product, the company designs the product. If the company is satisfied with the design, the company launches the product and the process ends. If not, the company redesigns the product. "
        "On the other hand, if the company chooses to modify an existing product, the company chooses a product to redesign and then redesigns the product.\n"
        "Exclusive gateway 2: If the company is satisfied with the design, the company launches the product and the process ends. If not, the company redesigns the product.\n\n"
        "Process: 'if approved, the client finalizes their project.'\n"
        "Exclusive gateway 1: if approved, the client finalizes their project.\n\n"
        "Process: 'If the student is rejected, the employer notifies the student. If the student is accepted, the professor notifies the student via email and Slack in parallel. "
        "The student then fills out the application form. The student hands in his internship journal. Finally, the professor updates the Airtable database and the process ends.'\n"
        "Exclusive gateway 1: If the student is rejected, the employer notifies the student. If the student is accepted, the professor notifies the student via email and Slack in parallel. The student then fills out the application form. "
        "The student hands in his internship journal. Finally, the professor updates the Airtable database and the process ends.\n\n"
        f"Process: '{process_description}'\n"
        "Exclusive gateway 1:"
    )

    # Формируем финальный промпт
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Токенизируем промпт
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Получаем только сгенерированную часть
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()


def extract_exclusive_gateways_2_conditions(process_description: str) -> str:
    """
    Извлекает задачи для указанного exclusive gateway по двум условиям,
    включая не-непосредственные задачи, используя deepseek-llm-7b-chat.
    """
    # Системное сообщение — общая инструкция
    system_msg = (
        "You are a highly experienced business process modelling expert, specializing in BPMN modelling. "
        "Extract the text belonging to a specific exclusive gateway, including all tasks that follow the gateway conditions."
    )

    # Примеры и описание задачи
    user_msg = (
        "You will receive a description of a process which contains conditions. "
        "Extract the text which belongs to a specific exclusive gateway. "
        "You have to extract all the tasks that follow from the conditions in the exclusive gateway, not only the immediate tasks.\n\n###\n\n"

        "Process: 'If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. "
        "If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase. "
        "Once the client has chosen to fund or pay with currency, they must sign the agreement before concluding the transaction.'\n"
        "Exclusive gateway 1: If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. "
        "If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase.\n\n"

        "Process: 'If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.'\n"
        "Exclusive gateway 1: If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.\n\n"

        "Process: 'If yes, the manager prepares additional questions. If the decision is not to prepare, the manager waits for the customer. "
        "After the paths merge, the customer sends the application.'\n"
        "Exclusive gateway 1: If yes, the manager prepares additional questions. "
        "If the decision is not to prepare, the manager waits for the customer.\n\n"

        "Process: 'If the student is rejected, the employer notifies the student. If the student is accepted, the professor notifies the student via email and Slack in parallel. "
        "The student then fills out the application form. The student hands in his internship journal. Finally, the professor updates the Airtable database and the process ends.'\n"
        "Exclusive gateway 1: If the student is rejected, the employer notifies the student. If the student is accepted, the professor notifies the student via email and Slack in parallel. "
        "The student then fills out the application form. The student hands in his internship journal. Finally, the professor updates the Airtable database and the process ends.\n\n"

        f"Process: '{process_description}'\n"
        "Exclusive gateway 1:"
    )

    # Собираем prompt через шаблон
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Токенизируем
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Получаем только ответ
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()



def extract_gateway_conditions(process_description: str, conditions: str) -> str:
    """
    Группирует условия по exclusive gateway'ам, используя deepseek-llm-7b-chat.
    Используются только переданные условия.
    """
    # Системный prompt
    system_prompt = (
        "You are an expert in business process modeling. You will be given a process description and a list of conditions. "
        "Your task is to group the conditions by the exclusive gateways they belong to. Only use the listed conditions. "
        "Format your answer like:\nExclusive gateway 1: <cond1> || <cond2>\nExclusive gateway 2: <cond3> || <cond4>\n..."
    )

    # Пользовательский prompt с примерами
    user_prompt = (
        "You will receive a description of a process and a list of conditions that appear in the process. "
        "Determine which conditions belong to which exclusive gateway. Use only the conditions that are listed, "
        "do not take anything else from the process description.\n\n###\n\n"

        "Process: 'The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, "
        "the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer "
        "will need to bring the total cost of the car to the dealership in order to complete the transaction.'\n"
        "Conditions: 'If the customer chooses to finance', 'If the customer chooses to pay in cash'\n"
        "Exclusive gateway 1: If the customer chooses to finance || If the customer chooses to pay in cash\n\n"

        "Process: 'The restaurant receives the food order from the customer. If the dish is not available, the customer is informed "
        "that the order cannot be fulfilled. If the dish is available and the payment is successful, the restaurant prepares and serves the order. "
        "If the dish is available, but the payment fails, the customer is notified that the order cannot be processed.'\n"
        "Conditions: 'If the dish is not available', 'If the dish is available and the payment is successful', 'If the dish is available, but the payment fails'\n"
        "Exclusive gateway 1: If the dish is not available || If the dish is available and the payment is successful || If the dish is available, but the payment fails\n\n"

        "Process: 'The customer places an order on the website. The system checks the inventory status of the ordered item. "
        "If the item is in stock, the system checks the customer's payment information. If the item is out of stock, the system sends an out of stock notification "
        "to the customer and cancels the order. After checking the customer's payment info, if the payment is authorized, the system generates an order confirmation and sends it "
        "to the customer, and the order is sent to the warehouse for shipping. If the payment is declined, the system sends a payment declined notification to the customer and cancels the order.'\n"
        "Conditions: 'If the item is in stock', 'If the item is out of stock', 'if the payment is authorized', 'If the payment is declined'\n"
        "Exclusive gateway 1: If the item is in stock || If the item is out of stock\n"
        "Exclusive gateway 2: if the payment is authorized || If the payment is declined\n\n"

        "Process: 'The process begins with the student choosing his preferences. Then the professor allocates the student. After that the professor notifies the student. "
        "The employer evaluates the candidate. If the student is accepted, the professor notifies the student. The student then completes his internship. "
        "If the student is successful, he gets a passing grade'\n"
        "Conditions: 'If the student is accepted','If the student is successful'\n"
        "Exclusive gateway 1: If the student is accepted\n"
        "Exclusive gateway 2: If the student is successful\n\n"

        f"Process: '{process_description}'\n"
        f"Conditions: {conditions}"
    )

    # Собираем prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Токенизация
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и обрезка промпта
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(prompt):].strip()

    return response


def extract_parallel_gateways(process_description: str) -> str:
    """
    Извлекает текст, относящийся к параллельным gateway'ам из описания бизнес-процесса, используя deepseek-llm-7b-chat.
    """
    # Системный prompt
    system_prompt = (
        "You are a highly experienced business process modelling expert, specializing in BPMN modelling. "
        "You will be provided with a description of a complex business process and will need to extract the text "
        "which belongs to a specific parallel gateway."
    )

    # Пользовательский prompt с примерами
    user_prompt = (
        "Process: 'The professor sends the mail to the student. In the meantime, the student prepares his documents.'\n"
        "Parallel gateway 1: The professor sends the mail to the student. In the meantime, the student prepares his documents.\n\n"
        "Process: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same. "
        "After the application has been approved, one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously. "
        "If both teams verify the information as accurate, the loan is approved and the process moves forward to the next step.'\n"
        "Parallel gateway 1: The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.\n"
        "Parallel gateway 2: one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously.\n\n"
        "Process: 'The process starts with the client discussing his ideas for the website. In the meantime, the agency presents potential solutions. "
        "After that, the developers start working on the project while the client meets with the representatives on a regular basis.'\n"
        "Parallel gateway 1: the client discussing his ideas for the website. In the meantime, the agency presents potential solutions.\n"
        "Parallel gateway 2: the developers start working on the project while the client meets with the representatives on a regular basis\n\n"
        "Process: 'The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.'\n"
        "Parallel gateway 1: The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.\n\n"
        "Process: 'The process starts when a group of chefs generate ideas for new dishes. At this point, 3 things occur in parallel: the first thing is the kitchen team analyzing the ideas for practicality. "
        "The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. "
        "The third path sees the accountants reviewing the potential cost of the dishes. Once each track has completed its analysis, the management reviews the findings of the analysis.'\n"
        "Parallel gateway 1: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. "
        "The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. "
        "The third path sees the accountants reviewing the potential cost of the dishes\n\n"
        "Process: 'The employee delivers the package. In the meantime, the customer pays for the service. Finally, the customer opens the package while the employee delivers the next package.'\n"
        "Parallel gateway 1: The employee delivers the package. In the meantime, the customer pays for the service.\n"
        "Parallel gateway 2: the customer opens the package while the employee delivers the next package.\n\n"
        "Process: 'The project manager defines the requirements. The process then splits into 2 parallel paths: in the first path the front-end development team designs the user interface. "
        "If the design is approved, the team implements it. If not, the team revises the design and continues to implement the approven parts of the design at the same time. "
        "In the second parallel path the back-end development team builds the server-side functionality of the app. After the two parallel paths merge, the QA team test the app's performance.'\n"
        "Parallel gateway 1: in the first path the front-end development team designs the user interface. If the design is approved, the team implements it. "
        "If not, the team revises the design and continues to implement the approven parts of the design at the same time. "
        "In the second parallel path the back-end development team builds the server-side functionality of the app.\n\n"
        "Process: 'The process starts with the student choosing his preference. In the meantime, the professor prepares the necessary paperwork. "
        "After that, the student starts his internship while the employer monitors the student's progress. Finally, the student completes his internship while the professor updates the database at the same time.'\n"
        "Parallel gateway 1: The student choosing his preference. In the meantime, the professor prepares the necessary paperwork.\n"
        "Parallel gateway 2: the student starts his internship while the employer monitors the student's progress.\n"
        "Parallel gateway 3: the student completes his internship while the professor updates the database at the same time.\n\n"
        "Process: 'The process starts when the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork. "
        "After that, the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.'\n"
        "Parallel gateway 1: the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork.\n"
        "Parallel gateway 2: the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.\n\n"
        f"Process: '{process_description}'"
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Преобразуем в формат модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Токенизация
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация текста
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодируем и обрезаем начало
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs):].strip()

    return response

def number_of_parallel_paths(parallel_gateway: str) -> str:
    """
    Определяет количество параллельных путей в описании параллельного шлюза (parallel gateway).
    Возвращает одно целое число в виде строки.
    """
    # Шаблон для запроса с примерами
    user_prompt = (
        "You will receive a description of a parallel gateway. Determine the number of parallel paths in the parallel gateway. "
        "Respond with a single number in integer format.\n\n###\n\n"
        "Parallel gateway: 'The R&D team researches and develops new technologies for the product. "
        "The next thing happening in parallel is the UX team designing the user interface and user experience. "
        "The interface has to be intuitive and user-friendly. The final thing occuring at the same time is when the QA team tests the product.'\n"
        "Number of paths: 3\n\n"
        "Parallel gateway: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.'\n"
        "Number of paths: 2\n\n"
        f"Parallel gateway: '{parallel_gateway}'\nNumber of paths:"
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_prompt}
    ]

    # Создаём input для модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и извлечение числа
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs.input_ids[0]):].strip()

    return response


def extract_parallel_tasks(sentence: str) -> str:
    """
    Извлекает задачи, выполняющиеся параллельно, из предложенной фразы и выводит их в указанном формате.
    """

    system_prompt = (
        "You are an expert in analyzing and structuring business process sentences. "
        "Given a sentence that includes multiple tasks being done in parallel, extract each task clearly. "
        "Output should follow the format:\n"
        "Task 1: <task>\nTask 2: <task>\n... as many tasks as necessary."
    )

    # Шаблон для запроса с примерами
    user_prompt = (
        'You will receive a sentence that contains multiple tasks being done in parallel.\n'
        'Extract the tasks being done in parallel in the following format (the number of tasks may vary):\n'
        'Task 1: <task>\nTask 2: <task>\n\n'
        '###\n\n'
        'Sentence: "The chef is simultaneously preparing the entree and dessert dishes."\n'
        'Task 1: prepare the entree\n'
        'Task 2: prepare the dessert dishes\n\n'
        'Sentence: "The project manager coordinates with the design team, the development team and the QA team concurrently."\n'
        'Task 1: coordinate with the design team\n'
        'Task 2: coordinate with the development team\n'
        'Task 3: coordinate with the QA team\n\n'
        f'Sentence: "{sentence}"'
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Создаём input для модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и извлечение ответа
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs.input_ids[0]):].strip()

    return response

def extract_3_parallel_paths(parallel_gateway: str) -> str:
    """
    Извлекает 3 параллельных пути из описания процесса и возвращает их в нужном формате.
    """

    system_prompt = (
        "You are an assistant that extracts three parallel paths from a process description. "
        "Return the spans in this format: <path> && <path> && <path>. Use '&&' only twice."
    )

    # Шаблон для запроса
    user_prompt = (
        "You will receive a process which contains 3 parallel paths.\n"
        "Extract the 3 spans of text that belong to the 3 parallel paths in the following format: <path> && <path> && <path>\n"
        "You must extract the entire span of text that belongs to a given path, not just a part of it.\n"
        "Use the && symbols only twice.\n\n"
        "###\n\n"
        "Process: the first thing is the kitchen team analyzing the ideas for practicality. "
        "The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. "
        "At the same time, the art team creates visual concepts for the potential dishes. "
        "The third path sees the accountants reviewing the potential cost of the dishes.\n"
        "Paths: the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe && "
        "the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes && "
        "the accountants reviewing the potential cost of the dishes\n\n"
        "Process: The R&D team researches and develops new technologies for the product. "
        "The next thing happening in parallel is the UX team designing the user interface and user experience. "
        "The interface has to be intuitive and user-friendly. The final thing occurring at the same time is when the QA team tests the product.\n"
        "Paths: The R&D team researches and develops new technologies for the product && "
        "the UX team designing the user interface and user experience && "
        "the QA team tests the product\n\n"
        f"Process: {parallel_gateway}\n"
        "Paths:"
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Создаём input для модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и извлечение ответа
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs.input_ids[0]):].strip()

    # Проверка, что модель вернула три пути
    assert "&&" in response and response.count("&&") == 2, "Model did not return 3 parallel paths"
    print("Parallel paths:", response, "\n")

    return response


def extract_2_parallel_paths(parallel_gateway: str) -> str:
    """
    Извлекает 2 параллельных пути из описания процесса и возвращает их в нужном формате.
    """

    system_prompt = (
        "You are a highly experienced business process modelling expert, specializing in BPMN modelling. "
        "You will be provided with a description of a complex business process which contains 2 parallel paths "
        "and will need to extract the 2 spans of text that belong to the 2 parallel paths in the following format: <path> && <path>. "
        "You must extract the entire span of text that belongs to a given path, not just a part of it. Use the && symbol exactly once."
    )

    # Шаблон для запроса
    user_prompt = (
        "Process: After that, he delivers the mail and greets people. Simultaneously, the milkman delivers milk.\n"
        "Paths: he delivers the mail and greets people && the milkman delivers milk\n\n"

        "Process: There are 2 main things happening in parallel: the first thing is when John goes to the supermarket. "
        "The second thing is when Amy goes to the doctor. Amy also calls John at the same time. After those 2 main things are done, John goes home.\n"
        "Paths: John goes to the supermarket && Amy goes to the doctor. Amy also calls John at the same time.\n\n"

        "Process: The team designs the interface. If it's approved, the team implements it. If not, the team revises the existing design "
        "and starts drafting a new one in parallel.\n"
        "Paths: the team revises the existing design && starts drafting a new one\n\n"

        "Process: in the first path the front-end development team designs the user interface. If the design is approved, "
        "the front-end development team implements it. If not, the front-end development team revises it and continues to implement "
        "the approven parts of the design at the same time. In the second parallel path the front-end development team builds the server-side functionality of the mobile app.\n"
        "Paths: the front-end development team designs the user interface. If the design is approved, the front-end development team implements it. "
        "If not, the front-end development team revises it and continues to implement the approven parts of the design at the same time. "
        "&& the front-end development team builds the server-side functionality of the mobile app.\n\n"

        "Process: the team designs the user interface. If the design is approved, the team implements the design. "
        "If not, the team revises the design and continues to implement the approven parts of the design at the same time.\n"
        "Paths: the team revises the design && continues to implement the approven parts of the design\n\n"

        "Process: The process is composed of 2 activities done concurrently: the first one is the customer filling out a loan application. "
        "The second activity is a longer one, and it is composed of the manager deciding whether to prepare additional questions. "
        "If yes, the manager prepares additional questions. If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time. "
        "After both activities have finished, the customer sends the application.\n"
        "Paths: the customer filling out a loan application && the manager deciding whether to prepare additional questions. If yes, the manager prepares additional questions. "
        "If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time\n\n"

        "Process: If the decision is not to prepare, the manager waits for the customer. After that, the manager sends an email. "
        "While sending an email, the manager also reads a newspaper.\n"
        "Paths: the manager sends an email && the manager also reads a newspaper\n\n"

        f"Process: {parallel_gateway}\n"
        "Paths:"
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Создаём input для модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и извлечение ответа
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs.input_ids[0]):].strip()

    # Проверка, что модель вернула два пути
    assert "&&" in response and response.count("&&") == 1, "Model did not return 2 parallel paths"
    print("Parallel paths:", response, "\n")

    return response


def find_previous_task(task: str, previous_tasks: str) -> str:
    """
    Определяет наиболее вероятную прямую предшествующую задачу из списка возможных вариантов.
    """
    # Шаблон для запроса
    user_prompt = (
        f"Task: '{task}'\n\n"
        f"Choices:\n{previous_tasks}"
    )

    # Формируем сообщения
    messages = [
        {"role": "system", "content": "You are a highly experienced business process analyst. "
                                     "You will receive a description of a task and a list of possible preceding tasks. "
                                     "Your job is to determine which task is the most likely direct predecessor. "
                                     "Only return the exact text of the selected task from the choices."},
        {"role": "user", "content": user_prompt}
    ]

    # Создаём input для модели
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодирование и извлечение ответа
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output[len(inputs.input_ids[0]):].strip()

    print("Selected previous task:", response, "\n")
    return response