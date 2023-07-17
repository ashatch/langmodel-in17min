from transformers import AutoTokenizer

model_name = 'google/flan-t5-base' # hugging face model
tokenizer = AutoTokenizer.from_pretrained(model_name)

line = 'What colour is the undoubtedly beautiful sky?'
tokens = tokenizer.tokenize(line)

print (tokens)
