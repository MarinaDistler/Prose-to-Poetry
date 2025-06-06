import torch
import random
import numpy as np
import typing as tp
import os
import wandb
import shutil
import re
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
from torch.amp import autocast

from promts import get_train_prompt, system_instruction

def clean_responces(responces):
    for i in range(len(responces)):
        response[i] = re.sub(r"<rhyme[AB]>.*?</rhyme[AB]>", "", response[i])
        response[i] = re.sub(r'<\|(?:[sS]\d+|count\d+)\|>', '', response[i])
    return responces

def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        if parser is not None:
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

def start_wandb(name, project, config=None):
    wandb.login(key=os.environ['WB_TOKEN'].strip(), relogin=True)
    entity = os.environ.get('WANDB_ENTITY', None)
    if entity is None:
        wandb.init(
            project=project,
            name=name,
            config=config
        )
    else:
        wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity
        )




