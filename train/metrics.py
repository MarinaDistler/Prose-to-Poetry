from rhymetagger import RhymeTagger

rt = RhymeTagger()
rt.load_model(model='ru')  # Загрузка русской модели рифм

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

    total_possible = sum(len(group) for group in scheme_map)
    return correct_rhymes / total_possible if total_possible > 0 else 0.0

def compute_metrics(eval_preds, rhyme_schemes, tokenizer):
    predictions, _ = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    total_penalty = 0
    perfect_count = 0
    rhyme_scores = []

    for pred, rhyme_scheme in zip(decoded_preds, rhyme_schemes):
        lines = [line.strip() for line in pred.split("\n") if line.strip()]
        num_lines = len(lines)

        # Штраф за отклонение от 4 строк
        penalty = abs(num_lines - 4)
        total_penalty += penalty

        if num_lines == 4:
            perfect_count += 1

        rhyme_score = check_rhyme_scheme(lines, scheme=rhyme_scheme)
        rhyme_scores.append(rhyme_score)

    total = len(decoded_preds)
    avg_penalty = total_penalty / total if total > 0 else 0.0
    avg_rhyme = np.mean(rhyme_scores) if rhyme_scores else 0.0

    return {
        "avg_line_count_penalty": avg_penalty,       # чем меньше, тем лучше
        "perfect_4_line_ratio": perfect_count / total,
        "avg_rhyme_accuracy_ABAB": avg_rhyme         # от 0 до 1, чем выше — тем лучше
    }

def make_metric_fn(rhyme_schemes, tokenizer):
    return lambda x: compute_metrics(x, rhyme_schemes, tokenizer)