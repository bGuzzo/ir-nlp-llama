"""
This script implements a command-line interface (CLI) for a conversational AI chatbot.
The chatbot is powered by a fine-tuned version of the LLaMAntino 3 8B model and is designed to engage in
conversations with users through the terminal.

The script manages the conversation history using the LLaMAntino 3 chat template, which includes a system
prompt to define the chatbot's personality and behavior. User inputs are continuously read from the command
line, and the model's responses are generated and displayed in real-time.

Key Features:
-   Interactive CLI for engaging with the chatbot.
-   Powered by the fine-tuned Formal-LLaMAntino-3 model.
-   Manages conversation history for context-aware responses.
-   Uses 4-bit quantization for efficient memory and processing.
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Model and Tokenizer Configuration ---

# Path to the local model directory.
base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3"

# Configuration for 4-bit quantization to optimize for memory and speed.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load the quantized model and its corresponding tokenizer.
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# --- System Prompt and Chat History ---

# The system prompt defines the personality, objectives, and constraints of the chatbot.
# It is the first message in the chat template and guides the model's behavior throughout the conversation.
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

# The `messages` list stores the conversation history, which is passed to the model
# to maintain context. It is initialized with the system prompt.
messages = [
    {"role": "system", "content": sys},
]

# --- Text Generation Pipeline ---

# Create a text generation pipeline using the loaded model and tokenizer.
pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  # Set to False as Langchain expects only the generated text.
    task='text-generation',
    max_new_tokens=128,  # Limit the number of generated tokens for efficiency.
    temperature=0.3,  # A lower temperature produces more deterministic and less creative responses.
    do_sample=True,
    top_p=0.95,
)

# --- Main Conversation Loop ---

while True:
    # Read user input from the command line.
    user_prompt = input("Q:\t")
    # Append the user's message to the conversation history.
    messages.append({"role": "user", "content": user_prompt})
    # Generate a response from the model.
    gen_seqs = pipe(messages)
    # Concatenate the generated sequences into a single response string.
    segn_seq_str = "".join([seq['generated_text'] for seq in gen_seqs])
    # Append the model's response to the conversation history.
    messages.append({"role": "assistant", "content": segn_seq_str})
    # Print the model's response to the console.
    print(f"A:\t{segn_seq_str}")
