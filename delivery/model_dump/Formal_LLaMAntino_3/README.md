---
base_model: swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA
library_name: peft
---

# Model Card for Formal_LLaMAntino_3

This model, `Formal_LLaMAntino_3`, is a fine-tuned version of the `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` model. It has been further adapted through a rigorous fine-tuning process on the Italian Wikipedia dataset to enhance its proficiency and stylistic formality in the Italian language.

## Model Details

### Model Description

The primary objective of this fine-tuning was to specialize the base model for generating formal and scientific discourse in Italian. The process employed Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), making it feasible to train on consumer-grade hardware.

- **Developed by:** Bruno Guzzo
- **Model type:** Causal Language Model
- **Language(s) (NLP):** Italian
- **License:** Inherits the license from the base model, `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA`.
- **Finetuned from model:** `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA`

## Uses

### Direct Use

This model is intended for direct use as a conversational agent, particularly for applications requiring a high degree of formality and extensive general knowledge in the Italian language. It is well-suited for chatbots, content generation, and question-answering systems within academic or professional contexts.

### Out-of-Scope Use

This model is not intended for generating informal or colloquial Italian. Its performance on creative writing tasks may be limited due to its specialization in formal discourse.

## Bias, Risks, and Limitations

As a result of its training on the Italian Wikipedia, the model may exhibit biases present in the dataset. The fine-tuning process has also specialized the model towards a formal and scientific language style, which may result in suboptimal performance on tasks requiring informal or creative language.

## How to Get Started with the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model = "/path/to/your/model_dump/Formal_LLaMAntino_3"
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
```

## Training Details

### Training Data

The model was fine-tuned on the `20231101.it` split of the `wikimedia/wikipedia` dataset, which comprises a comprehensive corpus of Italian text.

### Training Procedure

The model was fine-tuned using the `unsloth` library for efficient training. The process involved 4-bit quantization and Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

#### Training Hyperparameters

- **`max_seq_length`**: 32
- **`num_train_epochs`**: 1
- **`per_device_train_batch_size`**: 8
- **`gradient_accumulation_steps`**: 4
- **`learning_rate`**: 2e-4
- **`optim`**: "paged_adamw_8bit"
- **`lora_r`**: 32
- **`lora_alpha`**: 8
- **`use_rslora`**: True

## Evaluation

### Testing Data, Factors & Metrics

- **Testing Data**: The model was evaluated on a held-out portion of the Italian Wikipedia dataset for perplexity, and on Italian versions of the ARC and HellaSwag benchmarks for commonsense reasoning.
- **Metrics**: Perplexity, ROUGE-1

### Results

The model demonstrated a significant improvement in perplexity on the Italian Wikipedia dataset when compared to its base model. However, a degradation in performance was observed on the ARC and HellaSwag benchmarks. This is hypothesized to be a consequence of the model's stylistic shift towards a more formal and scientific mode of expression, which may not be optimally aligned with the nature of these particular benchmarks.

## Environmental Impact

- **Hardware Type:** Consumer-grade GPU (8GB VRAM)
- **Hours used:** ~50 hours
- **Carbon Emitted:** Not measured
