import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

max_seq_length = 32 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

print("Loading Model... \t\t START")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print("Loading Model... \t\t COMPLETE")

print("\nLoading Dataset... \t\t START")

dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
dataset = dataset.shuffle(seed=42)  # Set a seed for reproducibility
n_samples = int(len(dataset) * 1.0)
dataset = dataset.select(range(n_samples))


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 10,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 1,
        learning_rate = 2e-4, #smaller steps for DPO and ORPO - standard 2e-4 for finetune
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",#"adamw_torch", #"paged_adamw_8bit",#paged_adamw_32bit
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "wiki_model",
        do_eval=False,
        torch_empty_cache_steps=1,
        save_strategy="steps",
        save_total_limit=3,
        save_steps=100
    ),
)

print("\nLoading Dataset... \t\t COMPLETE")

print('Training... \t\t START')
trainer_stats = trainer.train()
print('Training... \t\t END')
