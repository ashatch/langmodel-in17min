from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)
input_embeddings = model.get_input_embeddings()

line = 'What colour is the undoubtedly beautiful sky?'

tokens = tokenizer(line, return_tensors="pt")
outputs = model.generate(**tokens, generation_config=config)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
