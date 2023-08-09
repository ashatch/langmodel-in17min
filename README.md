# Language Models For Software Developers in 17 Minutes

Output from a follow-along from [this video](https://www.youtube.com/watch?v=tL1zltXuHO8).

Each of the steps builds on the previous:

- step 1: tokenizing the input
- step 2: getting token ids from tokens
- step 3: showing the pytorch tensors
- step 4: showing how to get embeddings
- step 5: do some inference on some input (ask the model a question)
- step 6: build a simple REPL loop to ask the model many questions
- step 7: try a larger model


# How to

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python step1_tokens.py
deactivate
...


# Notes

Requirements were frozen:

```bash
pip freeze > requirements.txt
```
