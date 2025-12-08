# MISP: Multilingual Idiomatic Sentence Paraphrasing

MISP (Multilingual Idiomatic Sentence Paraphrasing) is a system designed to generate idiomatic paraphrases in three languages: Portuguese (PT), French (FR), and Romanian (RO).

For Portuguese, a synthetic paraphrase dataset was automatically generated using the Apertus-4B model, based on the Portuguese subset of the AStitchInglanguemodels dataset (citation needed).
A Qwen model was then finetuned on this synthetic data.

The resulting finetuned model was used for inference in Portuguese, and transferred directly to French and Romanian.

## Repository Structure

**`PT_FR_RO/`**: Contains the synthetic datasets and the prompt-formatted data used for Portuguese finetuning.

**`scripts/`**: Includes code for:
- generating Portuguese synthetic data
- finetuning the Portuguese model
- running inference in PT, FR, and RO

**`predictions/`**: Contains the model predictions for the shared task test sets.
