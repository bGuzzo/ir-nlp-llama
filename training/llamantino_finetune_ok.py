import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastLanguageModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
    # BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 32 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 4, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # lora
    #modules_to_save = ["embed_tokens"] # add if you want to perturbate embedding layers for a new language adaptation
)


# ########## LLAMA 3 CONVERSATION TEMPLATE EXAMPLE ###################
# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(example):
#     prompt_messages = example["messages"]
#     templ = (tokenizer.apply_chat_template(prompt_messages, tokenize=False)+"<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>","<|end_of_text|>")
#     templ = templ.replace("<|eot_id|>"," <|eot_id|> ").replace("<|begin_of_text|>","<|begin_of_text|> ")
#     example["text"] = templ
#     return example

# from datasets import load_dataset
# dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split = "train_sft")
# dataset = dataset.map(formatting_prompts_func, batched = False,)

dataset = load_dataset("stanfordnlp/imdb", split="train")
# Mini dataset
dataset = dataset.shuffle(seed=42).select(range(10))

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
        output_dir = "train_outputs",
        do_eval=False,
        torch_empty_cache_steps=1,
        # save_steps=10
    ),
)

print('TRAIN -- Start')
trainer_stats = trainer.train()
print('TRAIN -- End')

# new_model = model_name+"_SFT_adapters"
# trainer.model.save_pretrained(new_model)


# # Reload model in FP16 and merge it with LoRA weights
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.bfloat16,
#     device_map="balanced"
# )
# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()

# # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(new_model+"_final_ultra")
# model.save_pretrained(new_model+"_final_ultra", safe_serialization=True)