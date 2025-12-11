# MISP: Multilingual Idiomatic Sentence Paraphrasing

MISP (Multilingual Idiomatic Sentence Paraphrasing) is a system designed to generate idiomatic paraphrases in three languages: Portuguese (PT), French (FR), and Romanian (RO).

For Portuguese, a synthetic paraphrase dataset was automatically generated using the Apertus-8B-Instruct-2509 model, based on the Portuguese subset of the AStitchInLanguageModels dataset (Tayaar-Madabushi, 2021). Finally, Qwen3-4B-Instruct-2507 model was finetuned on this synthetic data.

For Georgian, a synthetic paraphrase dataset was automatically generated using the Apertus-8B-Instruct-2509 model, based on the Georgian data for subtask 1 of the Parseme shared task 2.0. This synthetic dataset was then filtered to only keep paraphrases having a BERTscore equal to or higher than 0.7 with the original sentence. Finally, Qwen3-4B-Instruct-2507 model was finetuned on this synthetic data.


The resulting finetuned model was used for inference in Portuguese, and transferred directly to French and Romanian.

## Repository Structure

**`PT_FR_RO/`**: Contains the synthetic datasets and the prompt-formatted data used for Portuguese finetuning.

**`KA/`**: Contains the synthetic datasets and the prompt-formatted data used for Georgian finetuning.

Each of this folder includes three subfolders:



1. **`dataset/`**: Contains the synthetic dataset and the predictions for the shared task test sets.

2. **`predictions/`**: Contains the model predictions for the shared task test sets.

3. **`scripts/`**: Includes code for:
- generating synthetic data (Portuguese and Georgian);
- finetuning the model (once for Portuguese and once for Georgian);
- running inference in PT, FR, and RO and in KA.

  
  

