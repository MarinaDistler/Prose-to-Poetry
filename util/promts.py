from time import sleep
import pandas as pd
import ast

system_instruction = '''Вы – талантливый поэт, создающий русскую поэзию. При преобразовании прозы в стихотворение соблюдайте следующие правила:\n''' \
                    '''1. Рифмовка: используйте заданную схему рифм.\n''' \
                    '''2. Размер: пишите в указанном метре и соблюдайте структуру чередования ударных и безударных слогов.\n''' \
                    '''3. Объём: стихотворение должно содержать ровно 4 строки.\n''' \
                    '''4. Содержание: сохраняйте ключевые образы, эмоции, детали описаний и действия из исходного текста.\n''' \
                    '''5. Выразительность: сделайте текст поэтичным и насыщенным.\n''' \
                    '''6. Формат: в ответе должно быть исключительно само стихотворение без комментариев.\n'''

system_instruction_generate = '''Вы – талантливый поэт, создающий русскую поэзию. При написании стихотворения соблюдайте следующие правила:\n''' \
                    '''1. Рифмовка: используйте заданную схему рифм.\n''' \
                    '''2. Размер: пишите в указанном метре и соблюдайте структуру чередования ударных и безударных слогов.\n''' \
                    '''3. Объём: стихотворение должно содержать ровно 4 строки.\n''' \
                    '''5. Выразительность: сделайте текст поэтичным и насыщенным.\n''' \
                    '''6. Формат: в ответе должно быть исключительно само стихотворение без комментариев.\n'''

system_instruction_inv = '''Вы – профессиональный литературный редактор, специалист по адаптации русской поэзии в художественную прозу. Преобразуйте данное стихотворение в прозаический текст, соблюдая следующие правила:\n''' \
        '''1. Сохранение смысла: передайте основное содержание, образы, идеи и настроение стихотворения.\n''' \
        '''2. Объём: длина полученного текста должна соответствовать объёму исходного стихотворения.\n''' \
        '''3. Грамотная стилистика: используйте литературный язык, избегая слишком сухого или официального тона.\n''' \
        '''4. Связность и плавность: стройте текст логично и последовательно, обеспечивая естественный поток мысли.\n''' \
        '''5. Передача эмоций: сохраняйте эмоциональную окраску, экспрессию и художественные детали оригинала.\n''' \
        '''6. Избегание ритма и рифмы: уберите поэтические конструкции, но при необходимости используйте выразительные средства прозы (метафоры, эпитеты и т. д.).\n''' \
        '''7. Формат: в ответе должно быть исключительно результат без комментариев.\n'''
            
meters = {
    'ямб': 'ямбический - чередуются ударные и безударные слоги, первый слог строки безударный',
    'iambos': 'ямбический - чередуются ударные и безударные слоги, первый слог строки безударный',
    'choreios': 'хорей - чередуются ударные и безударные слоги, первый слог строки ударный',
    'dolnik3': 'дольник - стихотворный размер с переменным количеством безударных слогов между ударными',
    'amphibrachys': 'амфибрахий - трехсложный размер, где ударение падает на второй слог',
    'anapaistos': 'анапест - трехсложный размер, где ударение падает на третий слог',
    'daktylos': 'дактиль - трехсложный размер, где ударение падает на первый слог',
    'dolnik2': 'дольник - стихотворный размер с переменным количеством безударных слогов между ударными',
}

short_meters = {
    'ямб': 'ямб',
    'iambos': 'ямб',
    'choreios': 'хорей',
    'dolnik3': 'дольник',
    'amphibrachys': 'амфибрахий',
    'anapaistos': 'анапест',
    'daktylos': 'дактиль',
    'dolnik2': 'дольник',
}

def get_prompt(text, scheme='ABAB', meter='ямб'):
    if text is None:
        return f'''Напиши четверостишие с параметрами:\n Рифмовка: {scheme}\n Размер: {meters[meter]}\n'''
    return f'''Преобразуй прозу в четверостишие с параметрами:\n Рифмовка: {scheme}\n Размер: {meters[meter]}\n Исходный текст: {text}'''

def get_train_prompt(text, scheme='ABAB', meter='ямб'):
    if text is None:
        return f'''Напиши четверостишие с параметрами:\n Рифмовка: {scheme}\n Размер: {short_meters[meter]}\n'''
    return f'''Преобразуй прозу в четверостишие с параметрами:\n Рифмовка: {scheme}\n Размер: {short_meters[meter]}\n Исходный текст: {text}'''

def use_model_batch(func, texts, from_id=0):
    answers = {}
    for i, (id, text) in enumerate(texts.items()):
        if i // 15 < from_id:
            continue
        if i % 15 == 0 and len(answers) != 0:
            yield pd.Series(answers)
            answers = {}
        answers[id] = func(text)
    if len(answers) != 0:
        yield pd.Series(answers)

def generate_model_answers(model_func, file_path='test_text.txt', from_id=0, to_id=100):
    inputs = {}
    with open(file_path, 'r') as file:
        for i, line in enumerate(file.readlines()):
            inputs[i] = line
    answers = []
    i = from_id
    for answer in use_model_batch(model_func, inputs, from_id=from_id):
        answer.to_csv(f'answers{i}.txt')
        answers.append(answer)
        print(i)
        i += 1
        if i == to_id:
            return pd.concat(answers)
    return pd.concat(answers)


def format_chat_template(row, tokenizer, generate=False, markup='stanzas'):
    if generate:
        promt = get_train_prompt(None, row['rhyme_scheme'], row['meter'])
        row_json = [
            {"role": "system", "content": system_instruction_generate},
            {"role": "user", "content": promt},
        ]
    else:
        promt = get_train_prompt(row['input'], row['rhyme_scheme'], row['meter'])
        row_json = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": promt},
        ]
    if markup is not None:
        row_json.append(
            {"role": "assistant", "content": '\n'.join(ast.literal_eval(row[markup])) + '\n'}
        )
    row['promt'] = promt
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row