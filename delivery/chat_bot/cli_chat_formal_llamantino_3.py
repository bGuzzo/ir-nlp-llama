"""
    This pyhton script implements a conversational AI chatbot using 
    our fine-tuned version of LLaMAntino 3 8B.
    
    We store the chat history relying on the LLaMAntino 3 chat template. 
    At a given time the model imput prompt include the system prompt and the chat history
    by storing user inputs and previous generated responses.
"""

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3" # Chnage this path to the local model dump
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Use 4-bit quantization for memory and time efficency
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Loads the model using HF libraries
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)

# Loads the tokenizer corresponding to the loaded model. 
tokenizer = AutoTokenizer.from_pretrained(base_model)

# System prompt used as 0-time message in the chat-template.
# Instrcut model personality and behavior
sys = """
Sei un an assistente AI per la lingua Italiana di nome Formal-LLaMAntino-3

Obiettivo:
    Rispondi nella lingua usata per la domanda in modo chiaro e semplice.
    Rispondi in modo molto breve e coinciso.
    Usa meno parole possibile.

Personalità:
    Sei gentile, educato e disponibile con gli utenti.
    Mantieni un tono amichevole e colloquiale, come se stessi parlando con un amico.
    
Esempi di conversazione:
    Utente:
        Ciao! Come stai?
    AI:
        Ciao! Sto bene, grazie. Come posso aiutarti oggi?
    Utente:
        Puoi scrivermi una breve poesia sulla primavera?
    AI: 
        Certo! Ecco una poesia sulla primavera:
        La primavera è arrivata,
        La natura si è risvegliata.
        Fiori colorati sbocciano,
        E gli uccellini cantano felici.

Ricorda:
    Non hai accesso a informazioni personali sugli utenti.
    Non puoi accedere o condividere informazioni in tempo reale, come notizie o previsioni del tempo.
    Non sei in grado di eseguire azioni nel mondo fisico.
"""

"""
    Store the conversion history in a list.
    This will be input object to the model, which will later be compiled using the chat-template 
    in a text prompt.
    There is three role for this chat: system, user and assistant
"""
messages = [
    {"role": "system", "content": sys},
]

# Use HF pipeline
pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # Langchain expects the full text
    task='text-generation',
    max_new_tokens=128, # Small number of max out tokens for time efficiency
    temperature=0.3,  # Temperature for more or less creative answers
    do_sample=True,
    top_p=0.95,
)

# Main Conversation Loop, read and ingest user prompt
while(True):
    user_prompt = input("Q:\t")
    messages.append({"role": "user", "content": user_prompt})
    gen_seqs = pipe(messages)
    segn_seq_str = ""
    for seq in gen_seqs:
        segn_seq_str = segn_seq_str + seq['generated_text']
    messages.append({"role": "assistant", "content": segn_seq_str})
    print(f"A:\t{segn_seq_str}")
