# Импорт библиотек
import os, torch, wandb, sys
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ModelTLite, ModelQwen
from promts import format_chat_template 
from util import print_options


def main(args):
    quantization = False
    if args.model == 't-lite':
        model = ModelTLite(quantization=quantization, path=args.checkpoint, generate=args.generate, markup=args.markup)
    elif args.model == 'qwen':
        model = ModelQwen(quantization=quantization, path=args.checkpoint, generate=args.generate, markup=args.markup)
    if args.checkpoint != '':
        model.save_for_inference(args.checkpoint)
        model.load_for_inference(args.checkpoint)
    
    eval_data = pd.read_csv(args.test_dataset)
    result = []

    for _, row in tqdm(eval_data.iterrows()):
        result.append(model.use(row['text'], row['rhyme_scheme'], row['meter'], clean=not args.not_clean))

    df = pd.DataFrame({args.name: result}, index=eval_data.index)
    df.to_csv(args.output_dir + f'{args.name}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--name', type=str, default='t-lite', help='Saves the name.csv file with one column: name')
    parser.add_argument('--test_dataset', type=str, default='dataset/prosa_test_text.csv', help='Path to test dataset')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/', help='Output directory for results')
    parser.add_argument('--model', type=str, default='t-lite', choices=['t-lite', 'qwen'], help="Model type: 't-lite' or 'qwen'")
    parser.add_argument('--generate', action='store_true', help='If set, runs poetry generation instead of prose-to-poetry conversion')
    parser.add_argument('--not_clean', action='store_true', help='If set, disables postprocessing (doesn`t clean the output from markup)')
    parser.add_argument('--markup', type=str, default='stanzas', choices=['rhyme_markup', 'stress_markup', 'stanzas','rhyme_stress_markup'], help='The used markup')

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