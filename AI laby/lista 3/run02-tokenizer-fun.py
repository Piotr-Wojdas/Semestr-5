##### LLM - tokenizer
# W tym pliku operujemy na tokenach w bezpośredni sposób.
# TODO: podaj ścieżkę do modelu:
model_path = "AI laby/lista 3/dialogpt"

# tym razem chcemy torcha, żeby mieć dostęp do jego podstawowych struktur
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = "right")

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cpu")

# generujemy zapytanie bezpośrednio z tokenów
# tokeny są w katalogu DialoGPT, w pliku vocab.json
# odszyfruj jakie jest to pytanie
inputs = torch.stack([torch.tensor([2061,   338,   534, 12507,  2057,    30,   220, 50256])])
inputs.to("cpu")

outputs = model.generate(inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
outputs_extracted = outputs[:, inputs.shape[-1]:][0]
answer = tokenizer.decode(outputs_extracted, skip_special_tokens=True)
print("Odpowiedź: {}".format(answer))

# A tutaj generujemy dodatkową odpowiedź bezpośrednio z tokenów
outputs_extracted = torch.tensor([40,  1101,   407,   845, 14720,   996,    13,   220, 50256])
answer = tokenizer.decode(outputs_extracted, skip_special_tokens=True)
print("Odpowiedź: {}".format(answer))



question = tokenizer.decode([2061, 338, 534, 12507, 2057, 30, 220, 50256], skip_special_tokens=True)
print(question)