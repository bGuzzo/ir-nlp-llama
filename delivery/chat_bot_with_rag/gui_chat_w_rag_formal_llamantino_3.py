"""
This script implements a graphical user interface (GUI) for a Retrieval-Augmented Generation (RAG) chatbot.
The chatbot uses a fine-tuned version of the LLaMAntino 3 8B model and retrieves relevant information from a
knowledge base to provide more informed and context-aware responses.

This script provides a user-friendly interface built with Tkinter, allowing users to interact with the chatbot,
adjust model parameters, and view the conversation history.

Key Features:
-   GUI for easy interaction with the RAG chatbot.
-   Integration with a text retriever to augment the model's knowledge.
-   Adjustable parameters for temperature, top-p, top-k, and max new tokens.
-   Chat history display and reset functionality.
"""

import gc
import tkinter as tk
from tkinter import scrolledtext, Label, Entry
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from text_retriever import get_retrieved_text

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

# --- GUI Implementation ---

root = tk.Tk()
root.title("Formal-LLaMAntino-3 Chatbot (with RAG)")

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

Label(root, text="New Max Token (Could affect performance):").pack()
new_token_entry = Entry(root)
new_token_entry.insert(0, "128")
new_token_entry.pack()

Label(root, text="Max RAG Words (0 to avoid RAG usage):").pack()
rag_max_words_entry = Entry(root)
rag_max_words_entry.insert(0, "32")
rag_max_words_entry.pack()

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

    try:
        # Get parameters from the GUI.
        temperature = float(temp_entry.get())
        top_p = float(top_p_entry.get())
        top_k = int(top_k_entry.get())
        new_token = int(new_token_entry.get())
        rag_max_words = int(rag_max_words_entry.get())

        # Retrieve relevant text using the RAG system.
        retrieved_text = get_retrieved_text(user_prompt=user_input, max_words=rag_max_words)
        
        # Construct the model prompt, including the retrieved context if available.
        rag_model_input = user_input
        if retrieved_text:
            logging.warning("Using retrieved text into the user prompt")
            rag_model_input = f"""
                Contesto: {retrieved_text}
                
                Utente: {user_input}
            """
        
        logging.warning(f"RAG User Prompt:\t{rag_model_input}")
            
        messages.append({"role": "user", "content": rag_model_input})
        
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