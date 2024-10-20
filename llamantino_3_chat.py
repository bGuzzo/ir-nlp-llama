import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
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

sys = """
    Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA (Advanced Natural-based interaction for the ITAlian language).
    Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo.
    
"""

messages = [
    {"role": "system", "content": sys},
]

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # langchain expects the full text
    task='text-generation',
    max_new_tokens=512, # max number of tokens to generate in the output
    temperature=0.6,  #temperature for more or less creative answers
    do_sample=True,
    top_p=0.9,
)

# print('Chat template', tokenizer.chat_template)

while(True):
    user_prompt = input("Q:\t")
    messages.append({"role": "user", "content": user_prompt})
    gen_seqs = pipe(messages)
    segn_seq_str = ""
    for seq in gen_seqs:
        segn_seq_str = segn_seq_str + seq['generated_text']
    messages.append({"role": "assistant", "content": segn_seq_str})
    print(f"A:\t{segn_seq_str}")
