# Импорт библиотек
import os, torch, wandb, sys
import argparse
import pandas as pd
from tqdm.auto import tqdm
from evaluate import load
import numpy as np

bertscore = load("bertscore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ModelTLite, ModelQwen
from promts import format_chat_template 
from util import print_options
from metrics import check_rhyme_scheme, get_meter_score_isolated

def filter_lines(text):
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        for ch in line:
            if ch.isalpha():
                lines.append(line)
                break
    return lines
    
def check_len(lines):
    if len(lines) == 4 or len(lines) == 8:
        return 1.
    return 0.

def eval_poetry(inputs, outputs):
    result = pd.DataFrame(columns=['BERTscore', 'rhyme_score', 'meter_score', 'meter_score_done', 'len_score'])
    for name, outputs_ in outputs.items():
        bertscore_ = bertscore.compute(predictions=outputs_, references=inputs['text'], lang="ru")
        rhyme_scores = []
        meter_scores = []
        len_scores = []
        for i, output in tqdm(enumerate(outputs_)):
            lines = filter_lines(output)
            len_scores.append(check_len(lines))
            lines = lines[:8]
            rhyme_scores.append(check_rhyme_scheme(lines, inputs.iloc[i]['rhyme_scheme']))
            meter_scores.append(get_meter_score_isolated(lines, inputs.iloc[i]['meter']))
            
        meter_scores = np.array(meter_scores)
        res = {
            'BERTscore': np.mean(bertscore_["f1"]),
            'rhyme_score': np.mean(rhyme_scores),
            'meter_score': np.mean(meter_scores),
            'meter_score_done': np.mean(~np.isnan(meter_scores2)),
            'len_score': np.mean(len_scores),
        }
        result.loc[name] = res
        print(name, res)
    return result
    

def main(args):
    inputs = pd.read_csv(args.test_dataset)
    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Папка '{args.input_dir}' не существует.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    outputs = {}
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(args.input_dir, filename)
            name = filename[:-4]
            outputs[name] = pd.read_csv(file_path)[name].values
    res = eval_poetry(inputs, outputs) 
    print(res)
    res.to_csv(args.output_dir + f"{'_'.join(outputs.keys())}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--test_dataset', type=str, default='dataset/prosa_test_text.csv')
    parser.add_argument('--input_dir', type=str, default='output/models_output/')
    parser.add_argument('--output_dir', type=str, default='output/')

    args, unknown1 = parser.parse_known_args()

    unknown_args = set(unknown1)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)
    print_options(args, parser)
    main(args)