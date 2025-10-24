"""
This script implements a graphical user interface (GUI) for a conversational AI chatbot.
The chatbot is powered by a fine-tuned version of the LLaMAntino 3 8B model.

This script provides a user-friendly interface built with Tkinter, allowing users to:
- Interact with the chatbot in a conversational manner.
- Adjust model parameters such as temperature, top-p, top-k, and max new tokens.
- View the ongoing conversation in a chat log.
- Reset the chat history to start a new conversation.

The script is designed to be a standalone application that demonstrates the capabilities of the fine-tuned model in a practical, interactive setting.
"""

import gc
import tkinter as tk
from tkinter import scrolledtext, Label, Entry
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

# --- Initial Memory and Logger Configuration ---

gc.collect()
torch.cuda.empty_cache()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Model and Tokenizer Loading ---

base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# --- System Prompt and Chat History ---

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

messages = [{"role": "system", "content": sys}]

# --- GUI Implementation ---

root = tk.Tk()
root.title("Formal-LLaMAntino-3 Chatbot")

# --- Parameter Input Fields ---

Label(root, text="Temperature:").pack()
temp_entry = Entry(root)
temp_entry.insert(0, "0.3")
temp_entry.pack()

Label(root, text="Top-p:").pack()
top_p_entry = Entry(root)
top_p_entry.insert(0, "0.9")
top_p_entry.pack()

Label(root, text="Top-k:").pack()
top_k_entry = Entry(root)
top_k_entry.insert(0, "50")
top_k_entry.pack()

Label(root, text="Max New Tokens:").pack()
new_token_entry = Entry(root)
new_token_entry.insert(0, "128")
new_token_entry.pack()

# --- Chat Log and Input Field ---

chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_log.pack(expand=True, fill="both")
chat_log.config(state=tk.DISABLED)

entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=3)
entry.pack(expand=True, fill=tk.X, side=tk.BOTTOM)

def send_message():
    """Handles sending a message to the chatbot and displaying the response."""
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        return
    entry.delete("1.0", tk.END)
    logger.info(f"New user message: {user_input}")
    
    messages.append({"role": "user", "content": user_input})
    try:
        # Get parameters from the GUI.
        temperature = float(temp_entry.get())
        top_p = float(top_p_entry.get())
        top_k = int(top_k_entry.get())
        new_token = int(new_token_entry.get())
        
        # Create the text generation pipeline.
        pipe = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            task="text-generation",
            max_new_tokens=new_token,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k
        )
        
        logger.info(f"Initialized HF pipeline with: temp={temperature}, top-p={top_p}, top-k={top_k}, new_token={new_token}")

        # Generate the response and update the chat log.
        gen_seqs = pipe(messages)
        response = "".join([seq["generated_text"] for seq in gen_seqs])
        messages.append({"role": "assistant", "content": response})
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You:\t{user_input}\n")
        chat_log.insert(tk.END, f"AI:\t{response}\n\n")
        chat_log.see(tk.END)
        chat_log.config(state=tk.DISABLED)
        
        logger.debug(f"Chat history: {messages}")
        
        # Clean up memory.
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Error: {e}\n")
        chat_log.config(state=tk.DISABLED)

entry.bind("<Return>", lambda event: send_message())

# --- GUI Buttons and Main Loop ---

tk.Button(root, text="Send", command=send_message).pack(side=tk.BOTTOM)

def reset_chat_history():
    """Resets the chat history."""
    global messages
    messages = [{"role": "system", "content": sys}]
    chat_log.config(state=tk.NORMAL)
    chat_log.delete("1.0", tk.END)
    chat_log.config(state=tk.DISABLED)
    
tk.Button(root, text="Reset Chat History", command=reset_chat_history).pack(side=tk.BOTTOM)

def on_closing():
    """Handles the window closing event."""
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()