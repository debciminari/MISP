#!/bin/user/python3

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
input_json = './AStitchInLanguageModels/Dataset/TaskIndependentData/pt_TaskIndependentData.json'
output_json = 'data_pt.json'
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
gen_cfg.max_new_tokens = max_new_tokens 

# ==== LOAD DATA

with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data['test'][0][0])

seen = set()
unique_examples = []

for group in data['test']:
    for row in group:
        if row[8] == 0:
            example = (row[1], row[11], row[3]) 

            if example not in seen:
                seen.add(example)
                unique_examples.append({
                    'idiom': example[0],
                    'sentence': example[1],
                    'definition': example[2]
                })

print(f'An example from the input data: {unique_examples[0]}.')
print(f'The input data contains {len(unique_examples)} examples.')   

# ==== EXTRACT IDIOMS

idiom_items = []

for entry in unique_examples:
    sentence = entry['sentence']
    idiom = entry['idiom']
    definition = entry['definition']

    idiom_items.append({
        'sentence': sentence,
        'idiom': idiom,
        'lang': 'português',
        'definition': definition
    })

# ==== RESUME

if os.path.exists(output_json):
    print('Resuming from existing output file...\n')
    with open(output_json, 'r', encoding='utf-8') as f:
        prev = json.load(f)

    for old, new in zip(prev, idiom_items):
        if 'paraphrases' in old:
            new['paraphrases'] = old['paraphrases']

items_to_process = [x for x in idiom_items if 'paraphrase' not in x]

print(f'Total idioms:{len(idiom_items)}.\n')
print(f'Already done: {len(idiom_items) - len(items_to_process)}.\n')
print(f'Remaining: {len(items_to_process)}.\n')

# ==== BUILD PROMPTS

prompt_template = """Abaixo, você encontrará instruções descrevendo uma tarefa, juntamente com algum contexto.
Escreva uma resposta que atenda corretamente à solicitação. Antes de responder, pense cuidadosamente sobre a pergunta para fornecer uma resposta precisa.

### Instruções:
Sua língua nativa é {lang}. A frase a seguir contém a expressão idiomática {idiom}, que aqui significa {definition}.
Forneça 3 paráfrases que mantenham o significado original, substituindo a expressão idiomática por uma reformulação. Reformule toda a frase de modo que as paráfrases sejam bem diferentes umas das outras, use sinônimos e altere a estrutura da frase. Seja criativo.
Apresente sua paráfrase da seguinte forma, sem adicionar mais comentários:
As paráfrases são:
1. <paráfrase>.
2. <paráfrase>.
3. <paráfrase>.

### Frase: {sentence}.

### Resposta:
"""

prompts = [
    prompt_template.format(
        lang=item['lang'],
        idiom=item['idiom'],
        sentence=item['sentence'],
        definition=item['definition']
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

# ==== IMPROVED PARSING

import re

def _clean(s):
    if s is None:
        return ""
    s = s.strip()
    # remove leading numbering or bullets accidentally captured
    s = re.sub(r'^\s*(?:\d+[\.\)\-]\s*)', '', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # strip surrounding quotes
    s = s.strip(' \'"`')
    # remove a single trailing period (but keep internal punctuation)
    s = re.sub(r'\.$', '', s)
    return s.strip()

def parse_output(text):
    paraphrase1 = paraphrase2 = paraphrase3 = ""
    if not text:
        return {'paraphrase1': paraphrase1, 'paraphrase2': paraphrase2, 'paraphrase3': paraphrase3}

    triple_patterns = [
        r"1[\.\)]\s*(.+?)(?:\r?\n)+\s*2[\.\)]\s*(.+?)(?:\r?\n)+\s*3[\.\)]\s*(.+?)(?:\r?\n|$)",
        r"1\s*[-]\s*(.+?)(?:\r?\n)+\s*2\s*[-]\s*(.+?)(?:\r?\n)+\s*3\s*[-]\s*(.+?)(?:\r?\n|$)",
        r"As\s+par[aá]frases\s+s(?:ã|ao)\s*[:\-]?\s*(?:\r?\n)+\s*1[\.\)]\s*(.+?)(?:\r?\n)+\s*2[\.\)]\s*(.+?)(?:\r?\n)+\s*3[\.\)]\s*(.+?)(?:\r?\n|$)",
    ]

    for pat in triple_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            paraphrase1 = _clean(m.group(1))
            paraphrase2 = _clean(m.group(2))
            paraphrase3 = _clean(m.group(3))
            return {'paraphrase1': paraphrase1, 'paraphrase2': paraphrase2, 'paraphrase3': paraphrase3}

    numbered = re.findall(r'(?m)^\s*\d+\s*[\.\)\-]\s*(.+)$', text)
    if len(numbered) >= 3:
        paraphrase1, paraphrase2, paraphrase3 = (_clean(numbered[0]), _clean(numbered[1]), _clean(numbered[2]))
        return {'paraphrase1': paraphrase1, 'paraphrase2': paraphrase2, 'paraphrase3': paraphrase3}

    header_match = re.search(r"(As\s+par[aá]frases\s+s(?:ã|ao)\s*[:\-]?\s*)(.+)$", text, re.IGNORECASE | re.DOTALL)
    if header_match:
        after = header_match.group(2).strip()
        # split on lines that look like items (starting with number or dash) otherwise split by newline and pick top 3 meaningful lines
        candidates = re.findall(r'(?m)^\s*(?:\d+[\.\)\-]|\-)\s*(.+)$', after)
        if len(candidates) < 3:
            # try any non-empty line
            candidates = [ln.strip() for ln in after.splitlines() if ln.strip()]
        if len(candidates) >= 3:
            paraphrase1, paraphrase2, paraphrase3 = (_clean(candidates[0]), _clean(candidates[1]), _clean(candidates[2]))
            return {'paraphrase1': paraphrase1, 'paraphrase2': paraphrase2, 'paraphrase3': paraphrase3}

    return {'paraphrase1': paraphrase1, 'paraphrase2': paraphrase2, 'paraphrase3': paraphrase3}


# ==== MAIN LOOP

idx = 0
batch_counter = 0

for i in tqdm(range(0, len(prompts), batch_size)):
    batch = prompts[i:i + batch_size]
    outputs = generate_batch(batch)

    for out in outputs:
        items_to_process[idx]['paraphrases'] = parse_output(out)
        idx += 1

    batch_counter += 1
    if batch_counter % save_every == 0:
        print('Saving partial progress...')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(idiom_items, f, indent=2, ensure_ascii=False)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(idiom_items, f, indent=2, ensure_ascii=False)

print('All done...')
