from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "chavinlo/gpt4-x-alpaca"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
