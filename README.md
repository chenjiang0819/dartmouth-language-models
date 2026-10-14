# Language Models

An NLP project spanning tweet preprocessing, vocabulary encoding, conditional
n-gram models and GPT-2 fine-tuning. Chen Jiang implemented the student language-
modelling work for Dartmouth ENGS108 Applied Machine Learning in Fall 2022.

## Implementation

- Manual unigram, bigram and trigram counts with sentence-boundary tokens.
- Maximum-probability generation, seeded tie-breaking and trigram-to-bigram backoff.
- A Hugging Face GPT-2 training workflow with text preparation, checkpoint saving,
  reload and generation. GPT-2 is the upstream model; the n-gram algorithms are
  implemented here.

The maintained code separates corpus processing, count estimation and generation.
Vocabulary IDs are deterministic, conditional probabilities sum to one, and
unseen contexts and end-of-sentence tokens terminate bounded generation.

## Run

Use Python 3.12 and a virtual environment:

```sh
pip install -r requirements.txt
USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -v
```

Open [demo.ipynb](demo.ipynb) for a small n-gram example. To fine-tune a locally
available GPT-2 checkpoint, pass a text file, model, tokenizer and output directory
to `gpt2_training.train`. Training uses full blocks of up to 128 tokens and
next-token labels; generation reloads the saved model and tokenizer locally.

## Checks and results

Eight tests cover conditional-count oracles, boundary handling, final-candidate
selection, deterministic backoff, empty input, lexical preprocessing and GPT-2
interfaces. A tiny randomly initialized GPT-2 completed two CPU Trainer steps,
then saved, reloaded and generated text offline. The demo also ran in a fresh
kernel. These bounded examples demonstrate algorithm and training-workflow
behavior; they do not measure pretrained language quality.

## 2026 maintenance

AI-assisted maintenance corrected candidate selection, string/ID context mismatches,
trigram normalization and sentence boundaries. New support code uses a lexical
token cleaner and explicitly constructed causal-training samples. Selected
student implementations remain in [historical/student_cells.py](historical/student_cells.py);
the runnable modules and demo include the 2026 changes.
