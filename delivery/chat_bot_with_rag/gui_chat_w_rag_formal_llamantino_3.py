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
from text_retriever import get_retrieved_text

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

Rispondi nella lingua usata per la domanda in modo chiaro e semplice.
Rispondi in modo molto breve e coinciso.
Usa meno parole possibile.

Sei gentile, educato e disponibile con gli utenti.

Fai parte si un sistema RAG usato come chatbot.
La sezione 'Utente' contiene il messaggio dell'utente. Può essere vuota o mancante.
La sezione 'Contesto' contiene informazioni addizionali. Può essere vuota o mancante.
Ignora la sezione 'Contesto' se non è inerente alla sezione 'Utente'.
Ignora la sezione 'Contesto' se contiene informazioni non utili per l'utente.
"""

messages = [{"role": "system", "content": sys}]

# --- GUI Implementation using Tkinter ---
root = tk.Tk()
root.title("Formal-LLaMAntino-3 Chatbot (with RAG)")

# --- Parameter Input Boxes ---
temp_label = Label(root, text="Temperature:")
temp_label.pack()
temp_entry = Entry(root)
temp_entry.insert(0, "0.6")  # Default value
temp_entry.pack()

top_p_label = Label(root, text="Top-p:")
top_p_label.pack()
top_p_entry = Entry(root)
top_p_entry.insert(0, "0.9")  # Default value
top_p_entry.pack()

top_k_label = Label(root, text="Top-k:")
top_k_label.pack()
top_k_entry = Entry(root)
top_k_entry.insert(0, "50")  # Default value.
top_k_entry.pack()

new_token_label = Label(root, text="New Max Token (Could affect performance):")
new_token_label.pack()
new_token_entry = Entry(root)
new_token_entry.insert(0, "128")  # Default value. 
new_token_entry.pack()

rag_max_words_label = Label(root, text="Max RAG Words (0 to avoid RAG usage):")
rag_max_words_label.pack()
rag_max_words_entry = Entry(root)
rag_max_words_entry.insert(0, "32")  # Default value.
rag_max_words_entry.pack()

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
    try:
        temperature = float(temp_entry.get())
        top_p = float(top_p_entry.get())
        top_k = int(top_k_entry.get()) 
        new_token = int(new_token_entry.get())
        rag_max_words = int (rag_max_words_entry.get())
        
        # Compose final prompt
        retrieved_text = get_retrieved_text(user_prompt=user_input, max_words=rag_max_words)
        """
            Redefine the model prompt by adding retrieved text in the user message.
            We specified this structre in the system prompt.
        """
        rag_model_input = user_input
        if retrieved_text:
            logging.warning("Using retrieved text into the user prompt")
            rag_model_input = f"""
                Contesto: {retrieved_text}
                
                Utente: {user_input}
            """
        
        logging.warning(f"RAG User Prompt:\t{rag_model_input}")
            
        # Add user message to chat history
        messages.append({"role": "user", "content": rag_model_input})
        
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