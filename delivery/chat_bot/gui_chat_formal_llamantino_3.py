import gc
import tkinter as tk
from tkinter import scrolledtext, Label, Entry
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import logging

# Clean VRAM cahce
gc.collect()
torch.cuda.empty_cache()

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3"  # Change this path
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

messages = [{"role": "system", "content": sys}]

# --- GUI Implementation using Tkinter ---
root = tk.Tk()
root.title("Formal-LLaMAntino-3 Chatbot")

# --- Parameter Input Boxes ---
temp_label = Label(root, text="Temperature:")
temp_label.pack()
temp_entry = Entry(root)
temp_entry.insert(0, "0.3")  # Default value
temp_entry.pack()

top_p_label = Label(root, text="Top-p:")
top_p_label.pack()
top_p_entry = Entry(root)
top_p_entry.insert(0, "0.9")  # Default value
top_p_entry.pack()

top_k_label = Label(root, text="Top-k:")
top_k_label.pack()
top_k_entry = Entry(root)
top_k_entry.insert(0, "50")  # Default value.  Set to a reasonable default.
top_k_entry.pack()

new_token_label = Label(root, text="Max New Tokens:")
new_token_label.pack()
new_token_entry = Entry(root)
new_token_entry.insert(0, "128")  # Default value.  Set to a reasonable default.
new_token_entry.pack()

# --- Text Box ---
chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_log.pack(expand=True, fill="both")
chat_log.config(state=tk.DISABLED)  # Make it initially read-only

entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=3)
entry.pack(expand=True, fill=tk.X, side=tk.BOTTOM)

def send_message():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        return
    entry.delete("1.0", tk.END)
    logger.info(f"New user message: {user_input}")
    
    # Add user message to chat history
    messages.append({"role": "user", "content": user_input})
    try:
        temperature = float(temp_entry.get())
        top_p = float(top_p_entry.get())
        top_k = int(top_k_entry.get()) 
        new_token = int(new_token_entry.get())
        
        # Re-intialize HF Pipeline
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
        
        logger.info(f"Intialized HF pipeline with: temp={temperature}, top-p={top_p}, top-k={top_k}, new_token={new_token}")

        gen_seqs = pipe(messages)
        response = "".join([seq["generated_text"] for seq in gen_seqs])
        messages.append({"role": "assistant", "content": response})
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You:\t{user_input}\n")
        chat_log.insert(tk.END, f"AI:\t{response}\n")
        chat_log.insert(tk.END, "\n")
        chat_log.see(tk.END)  # Scroll to bottom
        chat_log.config(state=tk.DISABLED)
        
        logger.debug(f"Chat history: {messages}")
        
        # Prevent OOM
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Error: {e}\n")
        chat_log.config(state=tk.DISABLED)

entry.bind("<Return>", lambda event: send_message())  # Enter key sends message

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.BOTTOM)

def reset_chat_history():
    global messages
    messages = [{"role": "system", "content": sys}]
    chat_log.config(state=tk.NORMAL)  # Make the chat log editable
    chat_log.delete("1.0", tk.END)  # Clear the chat log
    chat_log.config(state=tk.DISABLED) # Make it read-only again
    
send_button = tk.Button(root, text="Reset Chat History", command=reset_chat_history)
send_button.pack(side=tk.BOTTOM)

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()