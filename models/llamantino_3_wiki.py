import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os

print(os.curdir)

# base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
# base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/models/llamantino_wiki_all_1"
base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301"
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

# sys = (
#     "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA "
#     "(Advanced Natural-based interaction for the ITAlian language)."
#     " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."
# )
# sys = """
#     Sei un an assisCtente AI per la lingua Italiana di nome LLaMAntino-3 ANITA (Advanced Natural-based interaction for the ITAlian language).
#     Rispondi alle richieste degli utenti nel modo più conciso possibile, usando il minor numero di parole, senza sacrificare la chiarezza e la completezza dell'informazione.
#     Privilegia risposte dirette e puntuali, evitando giri di parole e informazioni superflue.
#     Limita le risposte a un massimo di 250 parole.
#     Utilizza elenchi puntati o numerati quando possibile per presentare informazioni in modo sintetico.
#     Evita di ripetere informazioni già presenti nella domanda.
#     Concentrati sull'essenziale, fornendo solo le informazioni strettamente necessarie per rispondere alla domanda.
# """

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

messages = [
    {"role": "system", "content": sys},
]

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  # langchain expects the full text
    task="text-generation",
    max_new_tokens=512,  # max number of tokens to generate in the output
    temperature=0.6,  # temperature for more or less creative answers
    do_sample=True,
    top_p=0.9,
)

# print('Chat template', tokenizer.chat_template)

# Chat
# while(True):
#     user_prompt = input("Q:\t")
#     messages.append({"role": "user", "content": user_prompt})
#     gen_seqs = pipe(messages)
#     segn_seq_str = ""
#     for seq in gen_seqs:
#         segn_seq_str = segn_seq_str + seq['generated_text']
#     messages.append({"role": "assistant", "content": segn_seq_str})
#     print(f"A:\t{segn_seq_str}")

while True:
    user_prompt = input("Q:\t")
    chat = [
        {"role": "system", "content": sys},
    ]
    chat.append({"role": "user", "content": user_prompt})
    llm_reponse = pipe(chat)
    full_reponse = ""
    for seq in llm_reponse:
        full_reponse = full_reponse + seq["generated_text"]
    print(f"A:\t{full_reponse}")
