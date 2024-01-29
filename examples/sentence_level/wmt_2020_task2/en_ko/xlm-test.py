from transformers import AutoModel, AutoTokenizer

model_name = "D:/xlm-roberta-large"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input = tokenizer("Hello world!", return_tensors="pt")

output = model(**input)

print(output)