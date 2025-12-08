# ==== import
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
# tokenizer.pad_token = tokenizer.eos_token
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

# ==== load dataset
with open('data_pt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = []


for object in data:
    list_para = list(object['paraphrases'].values())

    for i in range(len(list_para)):
        idx = i+1
        paraphrase = object['paraphrases'][f'paraphrase{idx}']
        sentence = object['sentence']
        lang = object['lang']
        idiom = object['idiom']

        prompt = f"""Abaixo você encontrará instruções descrevendo uma tarefa, juntamente com algum contexto.
        Escreva uma resposta que atenda corretamente à solicitação. Antes de responder, pense cuidadosamente sobre a pergunta para fornecer uma resposta precisa.

        ### Instruções:
        Sua língua nativa é {lang}. A frase a seguir contém a expressão idiomática {idiom}.
        Forneça uma paráfrase que mantenha o significado original, substituindo a expressão idiomática por uma reformulação. Reformule toda a frase de modo a usar sinônimos e a alterar a estrutura da frase. Seja criativo.
        Apresente sua paráfrase da seguinte forma, sem adicionar mais comentários:
        A paráfrase é:
        1. <paráfrase>.

        ### Frase: {sentence}.

        ### Resposta:
        A paráfrase é:
        1. {paraphrase}.
        """
        prompts.append(prompt)

with open('prompts/pt_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, indent=2)

with open('prompts/pt_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'The dataset has {len(data)} instances.')

# === create train | val | test sets
dataset = load_dataset('json', data_files='prompts/pt_dataset.json')
dataset = dataset['train']

splits = dataset.train_test_split(test_size=0.2, seed=42)
test_val = splits['test'].train_test_split(test_size=0.5, seed=42)

train_set = splits['train']
val_set = test_val['train']
test_set = test_val['test']


print(f"Length of train set: {len(train_set)}.")
print(f'First examples of the training data: {train_set[0]}.')
print(f"Length of validation set: {len(val_set)}.")
print(f'First examples of the validation data: {val_set[0]}.')
print(f"Length of test set: {len(test_set)}.")
print(f'First examples of the test data: {test_set[0]}.')

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
    output_dir='./qwen-pt',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    fp16=True,
    optim='paged_adamw_8bit',
    eval_strategy='epoch',
    remove_unused_columns=False
)

# metric = evaluate.load('accuracy')
# def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

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
lora_dir = './qwen-pt/lora_adapters'
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)

print(f"LoRA adapters saved to {lora_dir}")

# Optional: Save training configuration for reproducibility
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

print(f"Training configuration saved")

## FOR TEST
import sys
from evaluate import compute_diversities, extract_mwe_tokens
from bert_score import score
from itertools import compress

print('\n' + '='*80)
print('EVALUATING ON TEST SET')
print('='*80)

model.eval()
model.config.use_cache = True

predictions = []
references = []
original_texts = []
masks = []

for idx, example in enumerate(test_set):
    text = example['text']

    # Extract original sentence (between "### Sentence:" and "### Response:")
    sentence_start = text.find('### Frase:') + len('### Frase:')
    sentence_end = text.find('### Reposta:')
    original_sentence = text[sentence_start:sentence_end].strip().rstrip('.')

    # Extract ground truth paraphrase
    paraphrase_start = text.find("### Resposta:\nA paráfrase é:") + len("### Resposta:\nA paráfrase é:")
    ground_truth = text[paraphrase_start:].strip()

    # Tokenize input (instruction part only, without the response)
    input_text = text[:sentence_end + len("### Resposta:\n")]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode prediction
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after "La paraphrase est :")
    if "A paráfrase é:" in full_output:
        prediction = full_output.split("A paráfrase é:")[-1].strip()
    else:
        prediction = full_output[len(input_text):].strip()
    
    predictions.append(prediction)
    references.append(ground_truth)
    original_texts.append(original_sentence)

    # Create a mock item to check if MWE was deleted
    # You'll need to parse idiom from your prompt
    idiom_start = text.find("expressão idiomática") + len("expressão idiomática")
    idiom_end = text.find(". Forneça uma paráfrase que")
    idiom = text[idiom_start:idiom_end].strip().rstrip('.')

    mock_item = {
        "text": f"[[{idiom}]]",  # Simplified - adjust based on your data structure
        "raw_text": original_sentence,
        "prediction": prediction
    }
    
    # Check if MWE tokens appear in prediction (basic check)
    idiom_tokens = idiom.split()
    prediction_lower = prediction.lower()
    mwe_deleted = not any(token.lower() in prediction_lower for token in idiom_tokens)
    masks.append(mwe_deleted)
    
    if idx < 3:  # Print first 3 examples
        print(f"\n--- Example {idx+1} ---")
        print(f"Original: {original_sentence}")
        print(f"Prediction: {prediction}")
        print(f"Reference: {ground_truth}")
        print(f"MWE deleted: {mwe_deleted}")

# ==== Compute BERTScore
print("\n" + "="*80)
print("BERTSCORE RESULTS")
print("="*80)

# Apply mask (only evaluate where MWE was deleted)
masked_predictions = list(compress(predictions, masks))
masked_references = list(compress(references, masks))

if len(masked_predictions) > 0:
    P, R, F1 = score(masked_predictions, masked_references, lang="fr", verbose=True)
    
    avg_precision = P.mean().item() * 100
    avg_recall = R.mean().item() * 100
    avg_f1 = F1.mean().item() * 100
    
    # Account for zero scores (where MWE wasn't deleted)
    zero_score_count = len(masks) - sum(masks)
    adjusted_f1 = (F1.sum().item() * 100) / len(masks)  # Include zeros in average
    
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 (masked): {avg_f1:.2f}")
    print(f"Average F1 (adjusted with zeros): {adjusted_f1:.2f}")
    print(f"Number of evaluated elements: {len(masked_predictions)}")
    print(f"Number of elements with 0 score (MWE not deleted): {zero_score_count}")
else:
    print("WARNING: No valid predictions (MWE was not deleted in any prediction)")

# ==== Compute Diversity Metrics
print("\n" + "="*80)
print("DIVERSITY METRICS")
print("="*80)

try:
    entropy_pred, variety_pred, balance_pred = compute_diversities(predictions, original_texts)
    entropy_ref, variety_ref, balance_ref = compute_diversities(references, original_texts)
    
    print(f"Predictions - Entropy: {entropy_pred:.3f}, Variety: {variety_pred:.0f}, Balance: {balance_pred:.3f}")
    print(f"References  - Entropy: {entropy_ref:.3f}, Variety: {variety_ref:.0f}, Balance: {balance_ref:.3f}")
except Exception as e:
    print(f"Error computing diversity: {e}")

    # After evaluation section
results_file = './qwen-fr/evaluation_results.json'
evaluation_results = {
    'bertscore': {
        'avg_f1': avg_f1,
        'adjusted_f1': adjusted_f1,
        'evaluated_count': len(masked_predictions),
        'zero_score_count': zero_score_count
    },
    'diversity': {
        'predictions': {
            'entropy': entropy_pred,
            'variety': variety_pred,
            'balance': balance_pred
        },
        'references': {
            'entropy': entropy_ref,
            'variety': variety_ref,
            'balance': balance_ref
        }
    }
}

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Evaluation results saved to {results_file}")
