from rhymetagger import RhymeTagger
import nltk
import os
import sys

current_dir = os.path.dirname(__file__)  # Папка, где лежит текущий скрипт
external_code_path = os.path.abspath(os.path.join(current_dir, '..', 'external_code', 'verslibre', 'py'))
sys.path.append(external_code_path)
from poetry.phonetic import Accents
from generative_poetry.metre_classifier import ErrorsTable, MetreClassifier, \
                PatternAnalyzer, StressPredictorAdapter, Markup

nltk.download('punkt_tab')

rt = RhymeTagger()
rt.load_model(model='ru')  # Загрузка русской модели рифм

proj_dir = 'external_code/verslibre'
tmp_dir = os.path.join(proj_dir, 'tmp')
data_dir = os.path.join(proj_dir, 'data')
models_dir = os.path.join(proj_dir, 'models')

accents = Accents()
accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
accents.after_loading(os.path.join(tmp_dir, 'stress_model'))

stress_predictor = StressPredictorAdapter(accents)


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

    for pred, rhyme_scheme in zip(texts, rhyme_schemes):
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


def find_errors_meter(markup):
    num_lines = len(markup.lines)
    errors_table = ErrorsTable(num_lines)
    for l, line in enumerate(markup.lines):
        for metre_name, metre_pattern in MetreClassifier.metres.items():
            line_syllables_count = sum([len(word.syllables) for word in line.words])

            # Строчки длиной больше border_syllables_count слогов не обрабатываем.
            if line_syllables_count > MetreClassifier.border_syllables_count or line_syllables_count == 0:
                continue
            error_border = 7
            if metre_name == "dolnik2" or metre_name == "dolnik3":
                error_border = 3
            if metre_name == "taktovik2" or metre_name == "taktovik3":
                error_border = 2
            pattern, strong_errors, weak_errors, analysis_errored = \
                PatternAnalyzer.count_errors(MetreClassifier.metres[metre_name],
                                             MetreClassifier._MetreClassifier__get_line_pattern(line),
                                             error_border)
            #print(MetreClassifier._MetreClassifier__get_line_pattern(line))
            if analysis_errored or len(pattern) == 0:
                errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern, True)
                continue
            #corrections = MetreClassifier._MetreClassifier__get_line_pattern_matching_corrections(line, l, pattern)[0]
            #accentuation_errors = len(corrections)
            #strong_errors += accentuation_errors
            errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
    return errors_table

def check_meter(lines, meter_name):
    score = 0.
    for line in lines:
        markup = Markup.process_text(line, stress_predictor)  
        errors = find_errors_meter(markup).data[meter_name]
        weak = errors[0].weak_errors
        strong = errors[0].strong_errors
        count_syllables = sum([len(word.syllables) for word in markup.lines[0].words])
        if count_syllables > 0:
            score += (strong + weak) / float(count_syllables)
            if strong + weak > count_syllables:
                print('error')
                print(line)
    return score / len(lines)

def get_meter(lines):
    markup = Markup.process_text(lines, stress_predictor)  
    errors = find_errors_meter(markup)
    return errors.get_best_metre()