# Lab 3 Starter - Text to Wheel Commands

Starter project for **Lab 3 - Introduction to Python: external libraries
(including AI)** of the FOSSBot4AI course.

## Goal

Build a small command-line tool in Python that reads natural-language
commands (one per line) and produces wheel motor speeds in JSON.

```
Input:  "do przodu"
Output: {"action": "forward", "wheels": {"left": 0.5, "right": 0.5}}
```

You will implement two classifiers and compare them:

1. **Classical ML** (`classifier_sklearn.py`) - TF-IDF + LogisticRegression,
   trained on a small labelled dataset of Polish commands.
2. **Pretrained embeddings** (`classifier_st.py`) - a multilingual
   sentence-transformer that needs no training of its own.

You will then run both classifiers on two test files - one with Polish
commands, one with commands in English, German and Spanish - and explain
the difference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The first run of the sentence-transformer classifier downloads the model
> (~120 MB). Subsequent runs use the cached copy.

## Project layout

```
data/
  training_commands.csv       75 labelled examples (15 per action)
  examples/
    basic.txt                 Polish commands (both classifiers should work)
    multilingual.txt          EN / DE / ES commands (only ST should work)

src/
  text_to_wheels.py           CLI entry point (provided)
  wheel_mapping.py            action -> wheel speeds (provided)
  classifier_sklearn.py       YOU FILL IN THE TODOs
  classifier_st.py            YOU FILL IN THE TODOs
```

## What to implement

Open `src/classifier_sklearn.py` and `src/classifier_st.py` and complete
the TODO blocks. Each file has 5-7 small TODOs explained inline.

## How to run

After completing the TODOs:

```bash
# Classical ML on the basic Polish input
python -m src.text_to_wheels \
    --input data/examples/basic.txt \
    --output outputs/sklearn_basic.json \
    --classifier sklearn

# Sentence-transformer on the multilingual input
python -m src.text_to_wheels \
    --input data/examples/multilingual.txt \
    --output outputs/st_multilingual.json \
    --classifier st
```

## Experiment

Run both classifiers on both example files (four combinations) and
inspect the JSON outputs. The expected pattern is:

| Input            | Classifier | Expected accuracy |
| ---------------- | ---------- | ----------------- |
| basic.txt        | sklearn    | high              |
| basic.txt        | st         | high              |
| multilingual.txt | sklearn    | low               |
| multilingual.txt | st         | high              |

Write 3-5 sentences explaining **why** the sklearn classifier fails on
`multilingual.txt` while the sentence-transformer model works.

## Licence

CC BY 4.0 - same as the lab instruction.
