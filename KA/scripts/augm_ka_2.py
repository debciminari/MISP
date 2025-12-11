#!/usr/bin/python3
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import os
import re
import random

# ==== CONFIG

model_name = 'swiss-ai/Apertus-8B-Instruct-2509'
batch_size = 8
max_input_length = 256  
max_new_tokens = 256     
save_every = 5  
input_json = 'KA.json'
output_json = 'data_ka_2.json'
num_samples = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== LOAD MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

gen_cfg = model.generation_config
gen_cfg.temperature = 0.7
gen_cfg.top_p = 0.95
gen_cfg.do_sample = True
gen_cfg.max_new_tokens = max_new_tokens  # Use the config variable

# ==== LOAD DATA

with open(input_json, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

if len(raw_data) > num_samples:
    random.seed(42)
    sampled_data = random.sample(raw_data, num_samples)
else:
    sampled_data = raw_data

print(f"Selected {len(sampled_data)} random entries for processing.")

# ==== EXTRACT IDIOMS

idiom_items = []

for entry in sampled_data:
    sentence = entry['text']
    mwes = entry.get('MWEs', {})

    for key, idiom in mwes.items():
        idiom_items.append({
            'sentence': sentence,
            'idiom': idiom,
            'lang': 'ქართული'
        })

# ==== RESUME

if os.path.exists(output_json):
    print('Resuming from existing output file...\n')
    with open(output_json, 'r', encoding='utf-8') as f:
        prev = json.load(f)

    for old, new in zip(prev, idiom_items):
        if 'structured_output' in old:
            new['structured_output'] = old['structured_output']

items_to_process = [x for x in idiom_items if 'structured_output' not in x]

print(f'Total idioms:{len(idiom_items)}.\n')
print(f'Already done: {len(idiom_items) - len(items_to_process)}.\n')
print(f'Remaining: {len(items_to_process)}.\n')

# ==== BUILD PROMPTS

prompt_template = """
### ინსტრუქციები:

თქვენ ქართული ენის მშობლიური ენა ხართ. შემდეგ წინადადებაში მოცემულია მრავალსიტყვიანი გამოთქმა "{idiom}". წინადადების ფორმულირება შემდეგი ნაბიჯების მიხედვით შეგიძლიათ:

- ყურადღებით წაიკითხეთ წინადადება.

- დაფიქრდით მრავალსიტყვიანი გამოთქმის მნიშვნელობაზე, როგორც ეს ამ კონტექსტში გამოიყენება. ახლა, თქვენი სიტყვებით დაწერეთ მრავალსიტყვიანი გამოთქმის მნიშვნელობა. ეს დაგეხმარებათ მთელი წინადადების მნიშვნელობის გაგებაში.

- დაფიქრდით მთელი წინადადების მნიშვნელობაზე. ახლა, თქვენი სიტყვებით დაწერეთ მთელი წინადადების მთავარი იდეა: გამოიყენეთ სინონიმები, შეცვალეთ წინადადების სტრუქტურა და გრამატიკული სტრუქტურა.

- შეადარეთ თქვენი გადაწერილი წინადადება ორიგინალურ წინადადებას, რათა დარწმუნდეთ, რომ მნიშვნელობა შენარჩუნებულია.

თქვენი პასუხის ფორმატირება ასე ხდება:
გამოთქმის „{idiom}“ მნიშვნელობაა:

პერიფრაზია არის:
1)

### წინადადება: {sentence}.

### პასუხი:
"""

prompts = [
    prompt_template.format(
        idiom=item['idiom'],
        sentence=item['sentence']

    )
    for item in items_to_process
]

# ==== GENERATION FUNCTION

def generate_batch(batch):
    enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)

    out = model.generate(
        **enc,
        generation_config=gen_cfg,
        max_length=max_input_length
    )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    cleaned = []
    for p, full in zip(batch, decoded):
        cleaned.append(full[len(p):].strip())

    return cleaned

# ==== PARSING

def parse_output(response):
    """
    Extract meaning and paraphrase from Georgian model output.
    
    Expected format (with or without bold markdown):
    მრავალსიტყვიანი გამოთქმების მნიშვნელობა: [meaning text]
    პერიფრაზია: [paraphrase text]
    """
    result = {
        'meaning': None,
        'paraphrase': None
    }

    meaning_match = re.search(r':\s*([^\.]*\.)', response)
    meaning = meaning_match.group(1)
    result['meaning'] = meaning

    para_match = re.search(r'(?:[^:]*:){2}\s*([^\.]*\.)', response)
    paraphrase = para_match.group(1)
    result['paraphrase'] = paraphrase
    
    return result

# ==== MAIN LOOP

idx = 0
batch_counter = 0

for i in tqdm(range(0, len(prompts), batch_size)):
    batch = prompts[i:i + batch_size]
    outputs = generate_batch(batch)

    for out in outputs:
        print(out)
        print('#######################################')
        items_to_process[idx]['structured_output'] = parse_output(out)
        idx += 1

    batch_counter += 1
    if batch_counter % save_every == 0:
        print('Saving partial progress...')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(idiom_items, f, indent=2, ensure_ascii=False)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(idiom_items, f, indent=2, ensure_ascii=False)

print('All done...')
