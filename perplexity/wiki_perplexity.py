import datetime
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import shutil
from datasets import load_dataset
import json
import gc

MODELS_TO_TEST = [
    {
        "base_model": "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
        "model_name": "LLaMAntino-3"
    },
    {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "Llama-3"
    },
    {
        "base_model": "/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301",
        "model_name": "checkpoint-57301"
    }
]

# Config params
# Sliding window with no overlap
max_length = 64
stride = 64

def compute_ppl(base_model: str, model_name: str):
    print(f"Initialize on model: {model_name}")
    print(f"Model HF path: {base_model}")
    print("=" * shutil.get_terminal_size().columns)
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
    print("=" * shutil.get_terminal_size().columns)
    print("\n")

    print("=" * shutil.get_terminal_size().columns)
    print("Loading dataset...")

    dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
    dataset = dataset.shuffle(seed=42)  # Set a seed for reproducibility
    n_samples = int(len(dataset) * 0.0001)
    dataset = dataset.select(range(n_samples))

    print("Dataset Loaded")
    print("=" * shutil.get_terminal_size().columns)

    print("=" * shutil.get_terminal_size().columns)
    print("Start tokenization...")

    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    print("Tokenization complete")
    print("=" * shutil.get_terminal_size().columns)

    seq_len = encodings.input_ids.size(1)

    print("=" * shutil.get_terminal_size().columns)
    print("Computing Perplexity...")

    nlls = []
    prev_end_loc = 0
    print(f"Range: {range(0, seq_len, stride)}")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda:0")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print("Complete!")
    print("=" * shutil.get_terminal_size().columns)

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")
    
    # Clean VRAM
    del model
    del tokenizer
    del dataset
    del encodings
    gc.collect()
    torch.cuda.empty_cache() 
    
    return ppl

result = []
for obj in tqdm(MODELS_TO_TEST):
    base_model = obj['base_model']
    model_name = obj['model_name']
    ppl = compute_ppl(model_name=model_name, base_model=base_model)
    print(f"Model: {model_name}, PPL: {ppl}")
    result.append({
        "model_name": model_name,
        "ppl": float(ppl)
    })
    

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"PPL_Wiki_IT_all_{formatted_time}.json"
with open(file_name, "w") as outfile:
    json.dump(result, outfile, ensure_ascii=False)