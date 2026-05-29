# fossbot-text-to-cmd

Starter project for the **FOSSBot4AI** course - a small Python program that
classifies short English movement commands ("go forward", "turn left", "stop", ...)
into discrete actions and maps each action to wheel speeds for the FOSSBot.

This repository is the starting point for two labs:

- **Lab 3 - Introduction to Python: external libraries (including AI).**
  Fill in the TODOs in `src/classifier_sklearn.py` (TF-IDF +
  LogisticRegression) and `src/classifier_st.py` (sentence-transformer cosine
  similarity). `src/wheel_mapping.py` is provided pre-filled - it is the
  lookup table from action label to wheel speeds.
- **Lab 4 - Dockerization + VSCode.** Containerise the working application by
  writing your own `Dockerfile` and `docker-compose.yml`, then run it from the
  command line and from a VSCode dev container.

## Layout

```
src/                          Python sources (classifiers + entry point text_to_wheels)
data/training_commands.csv    Labelled dataset for the sklearn classifier
data/examples/                Sample command files to classify
_solutions/                   Reference solutions: Lab 3 classifiers + Lab 4 Dockerfile, compose, devcontainer
requirements.txt              Python dependencies
```

## Quick run (after the Lab 3 TODOs are completed)

```bash
pip install -r requirements.txt
python -m src.text_to_wheels \
    --input data/examples/basic.txt \
    --output basic_sklearn.json \
    --classifier sklearn
```

Replace `--classifier sklearn` with `--classifier st` to use the
sentence-transformer variant.

## Licence

CC BY 4.0.
