"""
This script is the original fine-tuning script for the LLaMAntino 3 8B model, designed to adapt the model to specific conversational datasets.
It leverages the `unsloth` library for efficient training and memory management, making it possible to fine-tune large models on consumer-grade hardware.

This script demonstrates how to:
- Load a pre-trained model and tokenizer using 4-bit quantization.
- Apply Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).
- Prepare a dataset for supervised fine-tuning (SFT) using a specific chat template.
- Configure and run the training process using the `SFTTrainer` from the `trl` library.
- Save the trained LoRA adapters and merge them with the base model.

References:
- LLaMAntino 3 GitHub: https://github.com/marcopoli/LLaMAntino-3-ANITA
- Unsloth Library: https://github.com/unslothai/unsloth
"""

import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Set the visible CUDA device to the first GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Model and Training Configuration ---
max_seq_length = 8192  # Maximum sequence length. Unsloth supports RoPE scaling for longer sequences.
dtype = None  # Auto-detect data type. Use Float16 for older GPUs, Bfloat16 for Ampere+.
load_in_4bit = True  # Enable 4-bit quantization for memory savings.
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

# Load the pre-trained model and tokenizer with optimizations from the `unsloth` library.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Configure and apply PEFT with LoRA to the model.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank. Suggested values: 8, 16, 32, 64, 128.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,  # Dropout is supported but 0 is optimized.
    bias="none",  # Bias type. "none" is optimized.
    use_gradient_checkpointing="unsloth",  # Use unsloth's optimized gradient checkpointing.
    random_state=42,
    use_rslora=False,  # Rank-Stabilized LoRA.
    loftq_config=None,  # LoftQ configuration.
)

# --- Dataset Preparation ---

# LLaMA 3 chat template for formatting conversational data.
llama3_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA (Advanced Natural-based interaction for the ITAlian language). Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo. <|eot_id|> <|start_header_id|>user<|end_header_id|>

{} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>

{} <|eot_id|> <|end_of_text|>"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(example):
    """Formats a single example from the dataset into the LLaMA 3 chat template."""
    prompt_messages = example["messages"]
    # Apply the chat template and handle edge cases.
    templ = (tokenizer.apply_chat_template(prompt_messages, tokenize=False) + "<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>", "<|end_of_text|>")
    templ = templ.replace("<|eot_id|>", " <|eot_id|> ").replace("<|begin_of_text|>", "<|begin_of_text|> ")
    example["text"] = templ
    return example

# Load the UltraChat 200k dataset for supervised fine-tuning.
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
# Apply the formatting function to the dataset.
dataset = dataset.map(formatting_prompts_func, batched=False)

# --- Trainer Configuration and Execution ---

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,
    packing=False,  # Packing can speed up training for short sequences.
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

# Start the training.
trainer_stats = trainer.train()

# --- Model Saving and Merging ---

new_model = model_name + "_SFT_adapters"

# Save the LoRA adapters.
trainer.model.save_pretrained(new_model)

# Reload the base model in 16-bit precision and merge the LoRA weights.
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload the tokenizer and save the final merged model.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(new_model + "_final_ultra")
model.save_pretrained(new_model + "_final_ultra", safe_serialization=True)