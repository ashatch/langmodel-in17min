from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'google/flan-t5-base' # hugging face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
input_embeddings = model.get_input_embeddings()

line = 'What colour is the undoubtedly beautiful sky?'

tokens = tokenizer(line, return_tensors="pt")
token_ids = tokens['input_ids'][0]
our_embeddings = input_embeddings(token_ids)
print (our_embeddings)
print (our_embeddings.size())

# torch.Size([10, 768]) meaning each token has 768 dimensions to represent its meaning