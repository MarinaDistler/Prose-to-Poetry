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
from trl import SFTTrainer, setup_chat_format
from datasets import Dataset
import pandas as pd
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.t_lite import ModelTLite
from util.promts import format_chat_template 
from util.util import print_options, seed_everything, WandbLogger, ChatGenerationCallback


def train(model, tokenizer, datasets, peft_config, clean_eval_data, args):
    checkpoint = None if args.checkpoint == '' else args.checkpoint
    if checkpoint is not None:
        print(f'Use checkpoint {checkpoint}')
        run_name = f"{args.name_run}-from-{checkpoint}"
    else:
        run_name = f"{args.name_run}-{datetime.now().strftime('%m-%d-%H-%M')}"
    output_dir = args.output_dir + run_name
    logger = WandbLogger(name=run_name, project='Poetry')
    config = vars(args)
    logger.start_logging({key: config[key] for key in set(config.keys()) - {'name_run'}})
    logger.save(__file__)

    tokenizer.pad_token = tokenizer.eos_token

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=26,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=500,              # Сохранять каждые 500 шагов
        save_total_limit=1,          # Макс. число чекпоинтов (старые удаляются)
        load_best_model_at_end=True, # Загружать лучшую модель в конце
        metric_for_best_model="eval_loss",  # Критерий выбора лучшей модели
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        peft_config=peft_config, # сам адаптер, который создали ранее
        #max_seq_length=512,
        #dataset_text_field="chat",
        data_collator=data_collator, # был импортирован
        args=training_arguments,
        callbacks=[ChatGenerationCallback(tokenizer, clean_eval_data, output_dir)],
        #packing= False,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    return trainer

def main(args):
    seed_everything()

    model = ModelTLite(quantization=True)

    # LoRA config / адаптер 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    eval_data = pd.read_csv(args.test_dataset, index_col='Unnamed: 0').rename({'gigachat-max': 'input'}, axis=1)
    dataset = {
        'train': pd.read_csv(args.train_dataset, index_col='Unnamed: 0').rename({'gigachat': 'input'}, axis=1),
        'test': eval_data,
    }

    format_chat_template_ = lambda row: format_chat_template(row, model.tokenizer)
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
    path_to_save = args.output_dir + "/T-Lite-finetuned/"
    trainer.save_model(path_to_save)
    model.model.save_pretrained(path_to_save + 't_lite_finetune.model')
    model.tokenizer.save_pretrained(path_to_save + 't_lite_finetune_tokenizer')
    answers = generate_model_answers(
        lambda text: model.use(text, scheme='ABAB', meter='ямб'), 
        file_path=args.final_test_file, 
        from_id=0
    )
    answers.to_csv(path_to_save + 'fine_tune_t_lite.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train t-lite')
    parser.add_argument('--name_run', type=str, default='t-lite')
    parser.add_argument('--train_dataset', type=str, default='dataset/gigachat_all_stanzas_inv.csv')
    parser.add_argument('--test_dataset', type=str, default='dataset/gigachat_max_all_stanzas_inv.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--final_test_file', type=str, default='dataset/test_text.txt')
    parser.add_argument('--checkpoint', type=str, default='')

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