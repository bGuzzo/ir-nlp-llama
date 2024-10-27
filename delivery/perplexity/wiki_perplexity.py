"""
    This scripts evaluates the perplexity of different large language models (LLMs)
    on a sample of the Italian Wikipedia dataset.

    We use it to evaulte the effectiveness of the fine-tuning we applied to LLaMAntino 3
    using the Italian Wikipedia dataset.

    HF documentation regarding 'Perplexity of fixed-length models'
    https://huggingface.co/docs/transformers/perplexity
"""

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

# Models to be tested
MODELS_TO_TEST = [
    {
        "base_model": "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
        "model_name": "LLaMAntino-3",
    },
    {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct", 
        "model_name": "Llama-3"
    },
    {
        # Our model - fine-tuned version of LLaMAntino 3 on Italian Wikipedia Dataset
        "base_model": "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3",
        "model_name": "Formal_LLaMAntino_3",
    },
]

# Config params
# Sliding window with no overlap
max_length = 64  # Defines the maximum input sequence length for the model
stride = 64  # Sets the step size for sliding through the text (no overlap in this case)


def compute_ppl(base_model: str, model_name: str):
    print(f"Initialize on model: {model_name}")
    print(f"Model HF path: {base_model}")
    print("=" * shutil.get_terminal_size().columns)
    print("Start loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization for efficiency
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

    # Loads the Italian Wikipedia dataset, the same used to fine-tune our model (Wiki-LLaMAntino)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
    # Shuffles the dataset for randomness and selects a small sample (0.01%) for efficiency and time saving.
    dataset = dataset.shuffle(
        seed=42
    )  # Set a seed for reproducibility, avoid randomness
    n_samples = int(len(dataset) * 0.0001)
    dataset = dataset.select(range(n_samples))

    print("Dataset Loaded")
    print("=" * shutil.get_terminal_size().columns)

    print("=" * shutil.get_terminal_size().columns)
    print("Start tokenization...")

    # Concatenates each text field of the sampled dataset with double newline separators
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    print("Tokenization complete")
    print("=" * shutil.get_terminal_size().columns)

    seq_len = encodings.input_ids.size(1)

    print("=" * shutil.get_terminal_size().columns)
    print("Computing Perplexity...")

    nlls = []  # Store negative log-likelihood loss for each window
    prev_end_loc = 0

    # Iterates through the tokenized input using a sliding window approach
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # May be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda:0")

        # Creates target_ids by cloning input_ids and masking previous tokens
        # with -100 to focus the loss calculation on the current window.
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

    # Compute the average negative log-likelihood loss for each window and exponentiates it
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
    base_model = obj["base_model"]
    model_name = obj["model_name"]
    ppl = compute_ppl(model_name=model_name, base_model=base_model)
    print(f"Model: {model_name}, PPL: {ppl}")
    result.append({"model_name": model_name, "ppl": float(ppl)})


now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"PPL_Wiki_IT_all_{formatted_time}.json"
with open(file_name, "w") as outfile:
    json.dump(result, outfile, ensure_ascii=False)
