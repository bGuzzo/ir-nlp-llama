# As propose by LLaMAntino 3 authors we chose to use Language Model Evaluation Harness suit
# to evaulate our moden on famous benchmarks.
# Reference: https://github.com/EleutherAI/lm-evaluation-harness

# Orignal
lm_eval --model hf \
    --model_args pretrained="swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA,load_in_4bit=True" \
    --tasks hellaswag_it,arc_it \
    --device cuda:0 \
    --batch_size 1 \
    --limit 0.20 \
    --seed 42 \
    --output_path "/home/bruno/Documents/GitHub/ir-nlp-llama/evluation/plain_llamantino3_eval" \
    --log_samples 

# Our
lm_eval --model hf \
    # Change this path to the model dump
    --model_args pretrained="/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301,load_in_4bit=True" \
    --tasks hellaswag_it,arc_it \
    --device cuda:0 \
    --batch_size 1 \
    --limit 0.20 \
    --seed 42 \
    --output_path "/home/bruno/Documents/GitHub/ir-nlp-llama/evluation/plain_llamantino3_eval" \
    --log_samples 