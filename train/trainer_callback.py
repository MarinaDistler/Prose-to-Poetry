import torch
import wandb
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
from torch.amp import autocast

from util.promts import get_train_prompt, system_instruction

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False,  # Padding будет позже в коллаторе
    )
    return {
        "input_ids": tokens["input_ids"].squeeze(0),
        "attention_mask": tokens["attention_mask"].squeeze(0),
        "index": example["index"]  # сохраняем индекс строки
    }


class ChatGenerationCallback(TrainerCallback):
    def __init__(
                self, tokenizer, eval_dataset, output_dir, batch_size,
                show_examples=10, compute_metrics=None,
                max_new_tokens=512, generate=False
                ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.show_examples = show_examples
        self.output_dir = output_dir
        self.compute_metrics = compute_metrics
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = None  # будет установлен позже
        self.dataloader = None  # создадим в on_train_begin

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.device = model.device
        format_chat_template_ = lambda row: format_chat_template(row, self.tokenizer, self.pretrain, markup=None)
        self.eval_dataset = self.eval_dataset.apply(
            format_chat_template_, axis=1
        )
        dataset = Dataset.from_pandas(self.eval_dataset[['text']])
        tokenized_dataset = dataset.map(tokenize)
        self.dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not model:
            return

        to_table = []
        count = 0
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            index = batch["index"]  

            with autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            if count < self.show_examples:
                original_rows = [self.eval_dataset.loc[i] for i in index]
                for row, response in zip(original_rows, decoded_responses):
                    to_table.append({
                        "User Prompt": row["promt"],
                        "Generated": response,
                        "Ground Truth": '\n'.join(ast.literal_eval(row["stanzas"]))
                    })

            if self.compute_metrics:
                self.compute_metrics(
                    response, self.eval_dataset.loc[index, 'rhyme_scheme'], compute_result=False
                )

            count += len(index)
        
        if self.compute_metrics:
            metrics = self.compute_metrics(None, None, compute_result=True)
            metrics['eval/step'] = state.global_step
            wandb.log(metrics)

        # Логируем в W&B
        if to_table:
            table = wandb.Table(columns=["User Prompt", "Generated", "Ground Truth"])
            for item in to_table:
                table.add_data(item["User Prompt"], item["Generated"], item["Ground Truth"])
            
            wandb.log({
                f"predictions_step_{state.global_step}": table,
                "eval/step": state.global_step
            })

        # Сохраняем модель при каждой валидации
        checkpoint_dir = f"{self.output_dir}/step-{state.global_step}"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Чекпоинт сохранён в {checkpoint_dir}")