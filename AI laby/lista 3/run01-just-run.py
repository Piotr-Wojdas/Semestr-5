##### LLM - uruchomienie
# W tym pliku ładujemy LLM-a, zadajemy mu pytanie i otrzymujemy odpowiedź.
# TODO: podaj ścieżkę do modelu:
model_path = "AI laby/lista 3/dialogpt"

import torch
# istotne: nie trzeba importować torch-a w kodzie,
# ale bez niego albo innego backendu nie uda się zaimportować następnych klas

# importujemy klasy Auto - z lenistwa i dla wygody
# moglibyśmy zaimportować konkretne klasy
# odpowiadające konkretnemu modelowi i tokenizerowi
# ale nie musimy, więc tego nie robimy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tokenizer (do zamiany tekstu na tokeny i w drugą stronę)
# padding_side - od której strony dodajemy "puste" tokeny
# (jak tekst/zapytanie jest za krótkie)
# zależy od modelu, dialogpt powinien być z prawej
# ale z lewej też się nie obrazi
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = "right")
print("<< Tokenizer gotowy")

# Model (wysyłamy na procesor)
# uwaga, na typowo cpu idzie domyślnie, ale tutaj się upewniamy
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cpu")
print("<< Model gotowy")

# Zapytanie (wysyłamy na procesor)
# .encode - zamienia tekst na tokeny
# eos_token - tym kończymy zapytanie
# return_tensors - pytorch tensor - dla wygody dalszych operacji
query = input("Zapytanie: [enter - domyślne] ")
if query == "":
	query = "What's the weather like today? "
inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')
inputs.to("cpu")
print("<< Zapytanie gotowe")

# Odpowiedź
# max_length - ile maksymalnie mamy mieć tokenów
# pad_token_id - czym wypełniać puste miejsca
# (program będzie protestować, że pad = eos; ignorujemy to)
# skip_special_tokens - sprawdź samemu podając False
# .decode - zamienia tokeny na tekst
outputs = model.generate(inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
outputs_extracted = outputs[:, inputs.shape[-1]:][0]
answer = tokenizer.decode(outputs_extracted, skip_special_tokens=True)
print("<< Odpowiedź gotowa")
print("Odpowiedź: {}".format(answer))
