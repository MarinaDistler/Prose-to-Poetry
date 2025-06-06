from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
import re

from promts import get_train_prompt, get_prompt, system_instruction, system_instruction_generate

class BaseModel:
    def __init__(self, model_name, path, quantization=False, generate=False, markup='stanzas'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.quantization = quantization
        self.generate = generate
        if quantization: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        special_tokens = []
        if 'rhyme' in markup:
            special_tokens += ['<rhymeA>', '</rhymeA>', '<rhymeB>', '</rhymeB>']
        if 'stress' in markup:
            special_tokens += [f'<|S{i}|>' for i in range(1, 11)]
            special_tokens += [f'<|s{i}|>' for i in range(1, 11)]
        if markup == 'stress_markup_count':
            special_tokens += [f'<|count{i}|>' for i in range(1, 11)]
        if len(special_tokens) > 0:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
        if path != '':
            self.model = PeftModel.from_pretrained(self.model, path)
            self.model.enable_adapter_layers()
        self.model.cuda()

    def save_for_inference(self, path):
        self.model = self.model.merge_and_unload().to(torch.bfloat16)
        self.model.save_pretrained(os.path.join(path, 'merged'), safe_serialization=True)

    def load_for_inference(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(path, 'merged'), 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def use(self, text, scheme='ABAB', meter='ямб'):
        self.model.eval()
        system_instruction_ = system_instruction
        text_ = text
        if self.generate:
            system_instruction_ = system_instruction_generate
            text_ = None
        if self.quantization:
            messages = [
                {"role": "system", "content": system_instruction_},
                {"role": "user", "content": get_train_prompt(text_, scheme, meter)}
            ]
        else:
            messages = [
                {"role": "system", "content": system_instruction_},
                {"role": "user", "content": get_prompt(text_, scheme, meter)}
            ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = response.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
        response = clean_responces([responce])[0]
        return response

class ModelQwen(BaseModel):
    def __init__(self, quantization=False, path='', generate=False, markup='stanzas'):
        super().__init__(
            'Qwen/Qwen2.5-3B-Instruct', path, 
            quantization, generate, markup=markup)

class ModelTLite(BaseModel):
    def __init__(self, quantization=False, path='', generate=False, markup='stanzas'):
        super().__init__(
            "t-tech/T-lite-it-1.0", path, quantization, 
            generate, markup=markup)