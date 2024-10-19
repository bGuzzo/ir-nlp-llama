import datetime
import json
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os
import shutil

base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
model_name = "LLaMAntino-3"
# base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301"
# model_name = "checkpoint-57301"
top_p = 0.95
temp = 0.6
n_token_out = 256

print(f"Initialize on model: f{model_name}")
print(f"Model HF path: f{base_model}")

print("\n")
print("="*shutil.get_terminal_size().columns)
print("Start loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Model loaded successfully")
print("="*shutil.get_terminal_size().columns)
print("\n")

sys = (
    "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA "
    "(Advanced Natural-based interaction for the ITAlian language). "
    "Rispondi alle richieste degli utenti nel modo più conciso possibile, usando il minor numero di parole, senza sacrificare la chiarezza e la completezza dell'informazione. "
    "Privilegia risposte dirette e puntuali, evitando giri di parole e informazioni superflue. "
    "Utilizza elenchi puntati o numerati quando possibile per presentare informazioni in modo sintetico. "
    "Evita di ripetere informazioni già presenti nella domanda. "
    "Concentrati sull'essenziale, fornendo solo le informazioni strettamente necessarie per rispondere alla domanda. "
    "Evita di riportare note, voci aggiuntive e informazioni correlate. "
    "Limita la risposta a 100 parole. "
)

# test_prompts: list[str] = [
#     "Scrivi una breve storia per bambini in rima sulla vita di Leonardo da Vinci, usando un linguaggio semplice e fantasioso.",
#     "Scrivi un sonetto in stile petrarchesco che esprima il dolore per la perdita di una persona cara.",
#     "Scrivi un articolo di giornale che riassuma le principali teorie sull'origine dell'universo, presentando i diversi punti di vista in modo obiettivo e neutrale.",
#     "Spiega il concetto di 'entropia' a una persona che non ha conoscenze scientifiche, utilizzando un'analogia con la vita di tutti i giorni.",
#     "Descrivi le principali differenze tra la fisica classica e la fisica quantistica, evidenziando le implicazioni filosofiche di queste differenze."
# ]

test_prompts: list[str] = [
    "Scrivi una breve storia per bambini in rima sulla vita di Leonardo da Vinci, usando un linguaggio semplice e fantasioso.",
    "Scrivi un sonetto in stile petrarchesco che esprima il dolore per la perdita di una persona cara.",
    # "Scrivi un articolo di giornale che riassuma le principali teorie sull'origine dell'universo, presentando i diversi punti di vista in modo obiettivo e neutrale.",
    # "Spiega il concetto di 'entropia' a una persona che non ha conoscenze scientifiche, utilizzando un'analogia con la vita di tutti i giorni.",
    "Descrivi le principali differenze tra la fisica classica e la fisica quantistica, evidenziando le implicazioni filosofiche di queste differenze."
]

result = {}

for prompt in test_prompts:
    print("="*shutil.get_terminal_size().columns)
    print(f"Input prompt:\t{prompt}")
    pipe = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,  # langchain expects the full text
        task="text-generation",
        max_new_tokens=n_token_out,  # max number of tokens to generate in the output
        temperature=temp,  # temperature for more or less creative answers
        do_sample=True,
        top_p=top_p,
    )
    chat = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt}
    ]
    llm_response = pipe(chat)
    full_reponse = ""
    for seq in llm_response:
        full_reponse = full_reponse + seq["generated_text"]
    print(f"\nResponse:\t{full_reponse}")
    print("="*shutil.get_terminal_size().columns)
    result[prompt]=full_reponse


now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{model_name}_top-p_{top_p}_temp_{temp}_token_{n_token_out}_{formatted_time}.json"
with open(file_name, "w") as outfile: 
    json.dump(result, outfile, ensure_ascii=False)

print(f"\nSaved file: {file_name}")