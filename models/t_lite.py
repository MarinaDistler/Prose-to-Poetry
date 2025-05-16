from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from util.promts import get_prompt

class ModelTLite:
    def __init__(self, quantization=False):
        model_name = "t-tech/T-lite-it-1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.quantization = quantization
        if quantization: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                quantization_config=bnb_config,
            ).to('cuda:0')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
            ).to('cuda')

    def use(self, text, scheme='ABAB', meter='ямб'):
        prompt = get_prompt(text, scheme, meter)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

