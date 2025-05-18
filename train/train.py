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
from trl import SFTTrainer, setup_chat_format, SFTConfig
from datasets import Dataset
import pandas as pd
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models import ModelTLite, ModelQwen
from util.promts import format_chat_template 
from util.util import print_options, seed_everything, start_wandb, ChatGenerationCallback


def train(model, tokenizer, datasets, peft_config, clean_eval_data, args):
    checkpoint = None if args.checkpoint == '' else args.checkpoint
    args.name_run = args.name_run if args.name_run != '' else args.model
    if checkpoint is not None:
        print(f'Use checkpoint {checkpoint}')
        run_name = f"{args.name_run}-from-{checkpoint}"
    else:
        run_name = f"{args.name_run}-{datetime.now().strftime('%m-%d-%H-%M')}"
    output_dir = args.output_dir + run_name + ('-pretrain' if args.pretrain else '')
    config = vars(args)
    project = 'Poetry-pretrain' if args.pretrain else 'Poetry'
    start_wandb(
        run_name, project=project, 
        config={key: config[key] for key in set(config.keys()) - {'name_run'}}
    )
    print_options(args, None)

    tokenizer.pad_token = tokenizer.eos_token

    if args.model == 't-lite':
        fact_bach_size = 4
    else:
        fact_bach_size = 8

    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=fact_bach_size,
        per_device_eval_batch_size=fact_bach_size,
        gradient_accumulation_steps=args.batch_size // fact_bach_size,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        eval_strategy="no" if args.pretrain else 'steps',
        eval_steps=args.save_steps // args.batch_size,
        logging_steps=args.log_steps // args.batch_size,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        group_by_length=True,
        report_to="wandb",
        save_strategy='steps',
        save_steps=args.save_steps // args.batch_size,              # Сохранять каждые 500 шагов
        save_total_limit=10 if args.pretrain else 2,          # Макс. число чекпоинтов (старые удаляются)
        load_best_model_at_end=False if args.pretrain else True, # Загружать лучшую модель в конце
        metric_for_best_model="eval_loss",  # Критерий выбора лучшей модели
        max_seq_length=512,
        packing= False,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    callbacks = [ChatGenerationCallback(tokenizer, clean_eval_data, output_dir)]
    if args.pretrain:
        datasets['test'] = None
        callbacks = None

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        peft_config=peft_config, # сам адаптер, который создали ранее
        #dataset_text_field="chat",
        data_collator=data_collator, # был импортирован
        args=training_arguments,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    return trainer

def main(args):
    seed_everything()

    if args.model == 't-lite':
        model = ModelTLite(quantization=True, path=args.from_pretrain)
    elif args.model == 'qwen':
        model = ModelQwen(quantization=True, path=args.from_pretrain)

    # LoRA config / адаптер 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    eval_data = pd.read_csv(args.test_dataset, index_col='Unnamed: 0')
    dataset = {
        'train': pd.read_csv(args.train_dataset, index_col='Unnamed: 0'),
        'test': eval_data,
    }

    format_chat_template_ = lambda row: format_chat_template(row, model.tokenizer, args.pretrain)
    dataset['train'] = dataset['train'].apply(
        format_chat_template_, axis=1
    )
    dataset['test'] = dataset['test'].apply(
        format_chat_template_, axis=1
    )
    dataset = {
        'train': Dataset.from_pandas(dataset['train'][['text']]),
        'test': Dataset.from_pandas(dataset['test'][['text']]),
    }

    trainer = train(model.model, model.tokenizer, dataset, peft_config, eval_data.iloc[:10], args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--name_run', type=str, default='')
    parser.add_argument('--train_dataset', type=str, default='dataset/trainset.csv')
    parser.add_argument('--test_dataset', type=str, default='dataset/testset.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--final_test_file', type=str, default='dataset/test_text.txt')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--model', type=str, default='t-lite', choices=['t-lite', 'qwen'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--pretrain', action='store_true', help='Если установлен, то запусткаентся претрайн на генерации стихов.')
    parser.add_argument('--from_pretrain', type=str, default='')

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