from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import sys

model_name = 'google/flan-t5-xl'

print ("Loading model " + model_name + "...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200, min_length=10)
input_embeddings = model.get_input_embeddings()

while True:
    line = input ("> ")
    tokens = tokenizer(line, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
