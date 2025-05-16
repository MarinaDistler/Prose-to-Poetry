import torch
import random
import numpy as np
import typing as tp
import os
import wandb
import shutil
from transformers import TrainerCallback
import ast

from util.promts import get_prompt, system_instruction

def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------\n'
    print(message)


def seed_everything(seed: int = 1729) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class WandbLogger:
    def __init__(self, name='base-name', project='Project-Poetry'):
        self.name = name
        self.project = project

    def start_logging(self, config=None):
        wandb.login(key=os.environ['WB_TOKEN'].strip(), relogin=True)
        entity = os.environ.get('WANDB_ENTITY', None)
        if entity is None:
            wandb.init(
                project=self.project,
                name=self.name,
                config=config
            )
        else:
            wandb.init(
                project=self.project,
                name=self.name,
                config=config,
                entity=entity
            )
        self.wandb = wandb
        self.run_dir = self.wandb.run.dir
        self.train_step = 0

    def log(self, scalar_name: str, scalar: tp.Any):
        self.wandb.log({scalar_name: scalar}, step=self.train_step, commit=False)

    def log_scalars(self, scalars: dict):
        self.wandb.log(scalars, step=self.train_step, commit=False)

    def next_step(self):
        self.train_step += 1

    def set_step(self, step):
        self.train_step = step

    def save(self, file_path, save_online=True):
        file = os.path.basename(file_path)
        new_path = os.path.join(self.run_dir, file)
        shutil.copy2(file_path, new_path)
        if save_online:
            self.wandb.save(new_path)

    def __del__(self):
        self.wandb.finish()





class ChatGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, output_dir, num_examples=10):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_examples = num_examples
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if not model:
            return

        results = []
        for i in range(min(self.num_examples, len(self.eval_dataset))):
            row = self.eval_dataset.iloc[i]
            
            user_prompt = get_prompt(row['input'], row['rhyme_scheme'], row['meter'])
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            results.append({
                "User Prompt": user_prompt,
                "Generated": response,
                "Ground Truth": '\n'.join(ast.literal_eval(row['stanzas']))
            })

        # Логируем в W&B
        if results:
            table = wandb.Table(columns=["User Prompt", "Generated", "Ground Truth"])
            for item in results:
                table.add_data(item["User Prompt"], item["Generated"], item["Ground Truth"])
            
            wandb.log({
                f"predictions_step_{state.global_step}": table,
                "eval/step": state.global_step
            })

        # Сохраняем модель при каждой валидации
        checkpoint_dir = f"{self.output_dir}/step-{state.global_step}"
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Чекпоинт сохранён в {checkpoint_dir}")