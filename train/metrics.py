from rhymetagger import RhymeTagger
import nltk
import torch

nltk.download('punkt_tab')

rt = RhymeTagger()
rt.load_model(model='ru')  # Загрузка русской модели рифм

def check_rhyme_scheme(lines, scheme="ABAB"):
    if len(lines) > 4:
        lines = lines[:4]
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
    return correct_rhymes / len(lines)

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

        rhyme_score += check_rhyme_scheme(lines, scheme=rhyme_scheme)

    avg_penalty = total_penalty 

    return {
        "eval/avg_line_count_penalty": avg_penalty,       # чем меньше, тем лучше
        "eval/perfect_4_line_ratio": perfect_count,
        "eval/avg_rhyme_accuracy": rhyme_score         # от 0 до 1, чем выше — тем лучше
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