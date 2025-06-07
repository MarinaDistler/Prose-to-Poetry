from rhymetagger import RhymeTagger
import nltk
import torch
import russian_scansion
import multiprocessing
import sys
import os
import time

nltk.download('punkt_tab')

rt = RhymeTagger()
rt.load_model(model='ru')  # Загрузка русской модели рифм

meter_names_to_russian = {
    "iambos": ('ямб', (0, 1)),
    "choreios": ('хорей', (1, 0)),
    "daktylos": ('дактиль', (1, 0, 0)),
    "amphibrachys": ('амфибрахий', (0, 1, 0)),
    "anapaistos": ('анапест', (0, 0, 1)),
}

def check_rhyme_scheme(lines, scheme="ABAB"):
    rhymes = rt.tag(lines, output_format=1)

    scheme_map = []
    for position in range(len(scheme)):
        scheme_map.append([])
        for i in range(len(scheme)):
            if i != position and scheme[i] == scheme[position]:
                scheme_map[position].append(i)

    correct_rhymes = 0
    for i, rhyme_group in enumerate(rhymes):
        scheme_group = scheme_map[i % len(scheme_map)]
        correct_rhymes += len(set(rhyme_group) & set(scheme_group))

    total_possible = len(lines)
    return correct_rhymes / total_possible if total_possible > 0 else 0.



def compute_metrics(texts, rhyme_schemes):
    total_penalty = 0
    perfect_count = 0
    rhyme_score = 0

    for pred, rhyme_scheme, meter in zip(texts, rhyme_schemes, meters):
        lines = [line.strip() for line in pred.split("\n") if line.strip()]
        num_lines = len(lines)

        # Штраф за отклонение от 4 строк
        penalty = abs(num_lines - 4)
        total_penalty += penalty

        if num_lines == 4:
            perfect_count += 1

        rhyme_score += check_rhyme_scheme(lines[:4], scheme=rhyme_scheme)

    avg_penalty = total_penalty 

    return {
        "eval/avg_line_count_penalty": avg_penalty,       # чем меньше, тем лучше
        "eval/perfect_4_line_ratio": perfect_count,
        "eval/avg_rhyme_accuracy": rhyme_score,         # от 0 до 1, чем выше — тем лучше
    }

class ComputeAggMetrics:
    def __init__(self):
        self.metrics = {}
        self.count = 0
        self.zero_metrics()
    
    def zero_metrics(self):
        self.metrics = {
            "eval/avg_line_count_penalty": 0.,       # чем меньше, тем лучше
            "eval/perfect_4_line_ratio": 0.,
            "eval/avg_rhyme_accuracy": 0.,
        }
        self.count = 0
    
    def __call__(self, texts, schemes, compute_result=False):
        if compute_result:
            result = {}
            for key in self.metrics:
                result[key] = self.metrics[key] / self.count
            self.zero_metrics()
            return result
        batch_metrics = compute_metrics(
            texts, schemes
        )
        for key, value in batch_metrics.items():
            self.metrics[key] += value
        self.count += len(texts)
        return None



def make_metric_fn():
    return ComputeAggMetrics()


def create_rpst():
    rpst = russian_scansion.create_rpst_instance('./models/RussianPoetryScansionTools/models')
    rpst.max_words_per_line = 100
    rpst.enable_dolnik = False
    return rpst

def get_meter_score(lines, meter, rpst):
    rpst.meters = [meter_names_to_russian[meter]]
    try: 
        scansion = rpst.align(lines)
        meter = meter_names_to_russian[meter][0]
        if scansion.meter != meter:
            print(f"external code returned meter {scansion.meter} instead of {meter}")
            return 0.
        return scansion.score
    except Exception as e:
        print(f"error in meter aligment: {e}")
        return 0.


# --- 1. Функция, которая будет выполняться в отдельном процессе ---
def _worker_get_meter_score(
    lines,
    meter_name_key,
    result_queue          # Очередь для передачи результата
):
    """
    Рабочая функция для запуска в отдельном процессе.
    Инициализирует rpst и выполняет расчеты для одной порции данных.
    """
    try:
        rpst_instance = create_rpst()            
        target_meter_data = meter_names_to_russian[meter_name_key]
        target_meter_russian_name = target_meter_data[0]
        
        rpst_instance.meters = [target_meter_data] # Установка целевого метра

        scansion = rpst_instance.align(lines)

        score = 0.0
        if scansion.meter != target_meter_russian_name:
            print(f"Worker PID {os.getpid()}: Внешний код вернул метр '{scansion.meter}' вместо '{target_meter_russian_name}'")
            score = 0.0
        else:
            score = scansion.score

        result_queue.put({"status": "success", "score": score})

    except Exception as e:
        print(f"Worker PID {os.getpid()}: Ошибка в расчете метра: {e}", file=sys.stderr)
        result_queue.put({"status": "error", "message": str(e), "error_type": type(e).__name__})


# --- 2. Функция-обертка для запуска в основном процессе ---
def get_meter_score_isolated(
    lines,
    meter,
    timeout_seconds=300 # Таймаут по умолчанию 5 минут
):
    """
    Запускает расчет оценки метра в отдельном процессе для изоляции памяти.
    """
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_worker_get_meter_score,
        args=(
            lines,
            meter,
            result_queue
        )
    )

    process.start()

    process.join(timeout=timeout_seconds)

    if process.is_alive():
        print(f"Main process: Дочерний процесс PID {process.pid} завис или не завершился вовремя ({timeout_seconds}с), принудительно завершаю.")
        process.terminate() # Принудительно завершаем
        process.join() # Ждем окончательного завершения
        return None
    elif not result_queue.empty():
        worker_output = result_queue.get()
        if worker_output["status"] == "success":
            return worker_output["score"]
        else:
            print(f"Main process: Дочерний процесс вернул ошибку: {worker_output['error_type']}: {worker_output['message']}", file=sys.stderr)
            return None
    else:
        # Это случай, когда процесс был KILLED, но не успел ничего вернуть.
        print(f"Main process: Дочерний процесс PID {process.pid} завершился, но не поместил ничего в очередь. Вероятно, был KILLED из-за нехватки памяти или другой системной проблемы.")
        return None