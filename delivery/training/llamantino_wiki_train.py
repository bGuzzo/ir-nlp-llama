# -*- coding: utf-8 -一体化-
"""
This script is dedicated to the fine-tuning of the LLaMAntino 3 8B model, a state-of-the-art language model,
on the comprehensive Italian Wikipedia dataset. The primary objective of this task is to enhance the model's
linguistic capabilities and performance specifically for the Italian language.

To make this computationally intensive process feasible on consumer-grade hardware, this script leverages a suite
of advanced optimization techniques. These include 4-bit quantization to reduce the model's memory footprint,
and Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to minimize the number of trainable
parameters.

This implementation is inspired by and builds upon the original fine-tuning script developed by the creators
of LLaMAntino 3, which is publicly available on their GitHub repository.

References:
- LLaMAntino 8B GitHub (Original Fine-Tuning Script): https://github.com/marcopoli/LLaMAntino-3-ANITA/blob/main/model_adaptation/finetune_llama3.py
- LLaMAntino 8B GitHub (Home): https://github.com/marcopoli/LLaMAntino-3-ANITA/tree/main
- LLaMAntino 8B Hugging Face Model Card: https://huggingface.co/swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Set the visible CUDA device to the first GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Model and Training Configuration ---

# Maximum sequence length for the model. Any input sequence longer than this will be truncated.
# This is a critical parameter that affects both memory usage and the model's ability to understand long contexts.
max_seq_length = 32

# Data type for model computations. `None` allows for automatic detection.
# Use 'Float16' for older GPUs like Tesla T4/V100, and 'Bfloat16' for newer Ampere architecture and above.
dtype = None

# Enable 4-bit quantization to significantly reduce memory usage. Set to False if not needed.
load_in_4bit = True

# Hugging Face model identifier for LLaMAntino 3.
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

print("Loading Model... \t\t START")

# Load the model and tokenizer using the FastLanguageModel class from the `unsloth` library,
# which is optimized for faster training and reduced memory consumption.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply Parameter-Efficient Fine-Tuning (PEFT) to the model.
# This technique, specifically using LoRA, drastically reduces the number of trainable parameters,
# making fine-tuning more accessible on limited hardware.
# Reference: https://huggingface.co/blog/peft
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Rank for LoRA. Suggested values are 8, 16, 32, 64, 128. Higher ranks mean more trainable parameters.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,  # Alpha parameter for LoRA, which scales the learned weights.
    lora_dropout=0,  # Dropout probability for LoRA layers. 0 is optimized.
    bias="none",  # Bias type for LoRA. "none" is optimized.
    use_gradient_checkpointing=True,  # Use gradient checkpointing to save memory, especially for long contexts.
    random_state=42,  # Seed for reproducibility.
    use_rslora=True,  # Use Rank-Stabilized LoRA for improved performance. Reference: https://huggingface.co/blog/damjan-k/rslora
    loftq_config=None,  # Configuration for LoftQ, another PEFT method.
)

print("Loading Model... \t\t COMPLETE")

print("\nLoading Dataset... \t\t START")

# Load the Italian Wikipedia dataset.
dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
# Shuffle the dataset to ensure random data distribution during training.
dataset = dataset.shuffle(seed=42)
# Use 100% of the dataset for fine-tuning.
n_samples = int(len(dataset) * 1.0)
dataset = dataset.select(range(n_samples))

# Configure and initialize the SFTTrainer for supervised fine-tuning.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,  # Number of processes for dataset preprocessing.
    packing=False,  # Set to True for faster training on short sequences.
    args=TrainingArguments(
        num_train_epochs=1,  # Train for one full epoch.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",  # Paged AdamW optimizer for memory efficiency.
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="wiki_model",
        do_eval=False,  # Disable evaluation during training to save time.
        torch_empty_cache_steps=1,
        save_strategy="steps",
        save_total_limit=3,  # Keep only the last 3 checkpoints.
        save_steps=100,  # Save a checkpoint every 100 steps.
    ),
)

print("\nLoading Dataset... \t\t COMPLETE")

print('Training... \t\t START')
# Start the training process.
trainer_stats = trainer.train()
print('Training... \t\t END')
