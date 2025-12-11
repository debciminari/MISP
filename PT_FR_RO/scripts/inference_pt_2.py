#!/bin/user/python3
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==== CONFIGURATION

BASE_MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'
LORA_ADAPTERS_PATH = './qwen-pt_2/lora_adapters'
INPUT_JSON_PATH = './test_sets/pt_test.blind.json'
OUTPUT_JSON_PATH = './preds/pt_preds_2-task.json'

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.8
BATCH_SIZE = 8

# ==== LOAD MODEL AND TOKENIZER

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = 'left'

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapters...")
model_with_adapter = PeftModel.from_pretrained(base_model, LORA_ADAPTERS_PATH)
model_with_adapter.eval()

print("Model loaded successfully!")

# ==== LOAD INPUT DATA

print(f"\nLoading data from {INPUT_JSON_PATH}...")
with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

print(f"Loaded {len(input_data)} sentences")

# ==== FUNCTION to CREATE PROMPT

def create_prompt(raw_text):
    return f"""
### Instruções:
Você é falante nativo de português brasileiro. Elabore uma reformulação da frase seguindo os passos abaixo:
- Leia a frase atentamente e certifique-se de compreender todo o seu significado.
- Escreva a ideia principal da frase com suas próprias palavras: use sinônimos, altere a estrutura da frase e a estrutura gramatical.
- Compare sua reformulação com a frase original para garantir que o significado foi preservado.
Formate sua resposta desta maneira:

A paráfrase é:
1)

### Frase: {raw_text}.

### Resposta:
    """

# ==== FUNCTION for INFERENCE

def generate_predictions_batch(texts, model, tokenizer, batch_size=BATCH_SIZE):
    all_predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        prompts = [create_prompt(text) for text in batch_texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            generated_text = tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            )
            all_predictions.append(generated_text.strip())

    return all_predictions


# ==== MAIN INFERENCE LOOP

print("\nStarting inference...")

raw_texts = [item['raw_text'] for item in input_data]
source_ids = [item['source_sent_id'] for item in input_data]

predictions = generate_predictions_batch(
    raw_texts,
    model_with_adapter,
    tokenizer,
    batch_size=BATCH_SIZE
)

results = [
    {
        'source_sent_id': source_id,
        'prediction': prediction
    }
    for source_id, prediction in zip(source_ids, predictions)
]

# ==== SAVE RESULTS

print(f"\nSaving results to {OUTPUT_JSON_PATH}...")
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Successfully saved {len(results)} predictions!")

# ==== POST-PROCESS: EXTRACT PARAPHRASE

def extract_span(prediction):
    pattern = r'(?<=1\))\s*([\s\S]*?)(?=\.|\n\s*2\))'
    patt_matching = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)

    if patt_matching:
        extracted = patt_matching.group(1).strip()
        return extracted

    return prediction

print("\nExtracting text spans...")
for result in results:
    result['prediction'] = extract_span(result['prediction'])

# ==== SAVE RESULTS

print(f"\nSaving results to {OUTPUT_JSON_PATH}...")
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Successfully saved {len(results)} predictions!")

# ==== DISPLAY SAMPLE RESULTS

print("\n" + "="*80)
print("SAMPLE PREDICTIONS (first 3):")
print("="*80)

for i, result in enumerate(results[:3], 1):
    print(f"\nSample {i}:")
    print(f"Source ID: {result['source_sent_id'][:80]}...")
    print(f"Prediction: {result['prediction'][:200]}...")
    print("-" * 80)
