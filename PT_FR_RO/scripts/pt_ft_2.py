#!/bin/user/python3
import json
from datasets import load_dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import evaluate
import numpy as np

from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig
)

# ==== load the model and the tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = 'Qwen/Qwen3-4B-Instruct-2507'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    )
model.config.pretraining_tp = 1
EOS_TOKEN = tokenizer.eos_token

# ==== set up lora configuration
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj'
    ]
)

model = get_peft_model(model, config).to(device)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f'Trainable params: {trainable_params:,} || all params: {all_param:,}.')

print_trainable_parameters(model)

# ==== load dataset
with open('data_pt_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = []

for object in data:
    list_para = list(object['paraphrases'].values())

    for i in range(len(list_para)):
        idx = i+1
        paraphrase = object['paraphrases'][f'paraphrase{idx}']
        sentence = object['sentence']
        idiom = object['idiom']

        prompt = f"""
### Instruções:
Você é falante nativo de português brasileiro. A frase a seguir contém a expressão composta "{idiom}". Elabore uma reformulação da frase seguindo os passos abaixo:
- Leia a frase atentamente e certifique-se de compreender todo o seu significado.
- Escreva a ideia principal da frase com suas próprias palavras: use sinônimos, altere a estrutura da frase e a estrutura gramatical.
- Compare sua reformulação com a frase original para garantir que o significado foi preservado.
Formate sua resposta desta maneira:

A paráfrase é:
1)

### Frase: {sentence}.

### Resposta:
1) {paraphrase}
"""

        prompts.append(prompt)

with open('prompts/pt_dataset_2.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, indent=2)

with open('prompts/pt_dataset_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'The dataset has {len(data)} instances.')

# === create train | val | test sets
dataset = load_dataset('json', data_files='prompts/pt_dataset_2.json')
dataset = dataset['train']

splits = dataset.train_test_split(test_size=0.2, seed=42)
test_val = splits['test'].train_test_split(test_size=0.5, seed=42)

train_set = splits['train']
val_set = test_val['train']
test_set = test_val['test']

# Save test set
test_data = test_set.to_list()

with open('test_set_".json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Length of train set: {len(train_set)}.")
print(f'First examples of the training data: {train_set[0]}.')
print(f"Length of validation set: {len(val_set)}.")
print(f'First examples of the validation data: {val_set[0]}.')
print(f"Length of test set: {len(test_set)}.")
print(f'First examples of the test data: {test_set[0]}.')
print('Saved test set.')

## FOR FINETUNING
# ==== load tokenizer
def tokenize_function(data):
    model_inputs = tokenizer(
        data['text'],
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors=None
        )

    model_inputs['labels'] = model_inputs['input_ids'].copy()

    return model_inputs

tokenized_train = train_set.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_val = val_set.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_test = test_set.map(tokenize_function, batched=True, remove_columns=['text'])

training_args = TrainingArguments(
    output_dir='./qwen-pt_2',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    fp16=True,
    optim='paged_adamw_8bit',
    eval_strategy='epoch',
    remove_unused_columns=False
)

# ==== data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==== trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    args=training_args,
    data_collator=data_collator
)

model.config.use_cache = False

trainer.train()
lora_dir = './qwen-pt_2/lora_adapters'
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)

print(f"LoRA adapters saved to {lora_dir}")

# Save training configuration
with open(f'{lora_dir}/training_config.json', 'w') as f:
    json.dump({
        'model_name': model_name,
        'num_epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'lora_r': config.r,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
    }, f, indent=2)

print(f"Training configuration saved.")
