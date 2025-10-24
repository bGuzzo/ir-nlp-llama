"""
This script evaluates the perplexity of various large language models (LLMs) on a subset of the Italian Wikipedia dataset.
Perplexity is a key metric for assessing the performance of a language model, as it measures how well the model predicts a given sequence of text.
A lower perplexity score indicates a better-performing model.

This script is primarily used to evaluate the effectiveness of the fine-tuning process applied to the LLaMAntino 3 model
by comparing its perplexity score against other models on the same dataset.

For more information on calculating perplexity for fixed-length models, refer to the Hugging Face documentation:
https://huggingface.co/docs/transformers/perplexity
"""

import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import shutil
from datasets import load_dataset
import json
import gc

# --- Model and Evaluation Configuration ---

# List of models to be evaluated.
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
        "base_model": "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/model_dump/Formal_LLaMAntino_3",
        "model_name": "Formal_LLaMAntino_3",
    },
]

# Configuration for the sliding window used in perplexity calculation.
max_length = 64  # Maximum input sequence length for the model.
stride = 64  # Step size for the sliding window. A stride equal to max_length means no overlap.

def compute_ppl(base_model: str, model_name: str):
    """Computes the perplexity of a given model on the Italian Wikipedia dataset."""
    print(f"Initialize on model: {model_name}")
    print(f"Model HF path: {base_model}")
    print("=" * shutil.get_terminal_size().columns)
    print("Start loading model...")

    # Configure 4-bit quantization for memory efficiency.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    # Load the model and tokenizer.
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

    # Load and prepare the Italian Wikipedia dataset.
    dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
    # Shuffle and select a small subset for efficiency.
    dataset = dataset.shuffle(seed=42)
    n_samples = int(len(dataset) * 0.0001)
    dataset = dataset.select(range(n_samples))

    print("Dataset Loaded")
    print("=" * shutil.get_terminal_size().columns)

    print("=" * shutil.get_terminal_size().columns)
    print("Start tokenization...")

    # Tokenize the dataset.
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    print("Tokenization complete")
    print("=" * shutil.get_terminal_size().columns)

    seq_len = encodings.input_ids.size(1)

    print("=" * shutil.get_terminal_size().columns)
    print("Computing Perplexity...")

    nlls = []  # To store negative log-likelihoods.
    prev_end_loc = 0

    # Iterate over the tokenized text with a sliding window.
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda:0")

        # Create target IDs, masking out tokens that are not part of the current window.
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print("Complete!")
    print("=" * shutil.get_terminal_size().columns)

    # Calculate perplexity as the exponential of the mean negative log-likelihood.
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")

    # Clean up memory.
    del model, tokenizer, dataset, encodings
    gc.collect()
    torch.cuda.empty_cache()

    return ppl


# --- Main Execution ---

result = []
for obj in tqdm(MODELS_TO_TEST):
    base_model = obj["base_model"]
    model_name = obj["model_name"]
    ppl = compute_ppl(model_name=model_name, base_model=base_model)
    print(f"Model: {model_name}, PPL: {ppl}")
    result.append({"model_name": model_name, "ppl": float(ppl)})

# Save the results to a JSON file.
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"PPL_Wiki_IT_all_{formatted_time}.json"
with open(file_name, "w") as outfile:
    json.dump(result, outfile, ensure_ascii=False)
