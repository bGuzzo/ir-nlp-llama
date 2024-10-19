lm_eval --model hf --model_args pretrained=HUGGINGFACE_MODEL_ID  --tasks hellaswag_it,arc_it  --device cuda:0 --batch_size auto:2
lm_eval --model hf --model_args pretrained=HUGGINGFACE_MODEL_ID  --tasks m_mmlu_it --num_fewshot 5  --device cuda:0 --batch_size auto:2 


# Optimize command
lm_eval --model hf --model_args pretrained="/home/bruno/Documents/GitHub/ir-nlp-llama/models/llamantino_wiki_all_1,load_in_4bit=True" --tasks hellaswag_it --device cuda:0 --batch_size 1 --limit 0.025 --seed 42 --output_path "/home/bruno/Documents/GitHub/ir-nlp-llama/lm-evaluation-harness/lm_eval_result/"  --log_samples

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
    --model_args pretrained="/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301,load_in_4bit=True" \
    --tasks hellaswag_it,arc_it \
    --device cuda:0 \
    --batch_size 1 \
    --limit 0.20 \
    --seed 42 \
    --output_path "/home/bruno/Documents/GitHub/ir-nlp-llama/evluation/plain_llamantino3_eval" \
    --log_samples 

# Not working
# lm_eval --model hf \
#     --model_args pretrained="swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA,load_in_4bit=True" \
#     --tasks m_mmlu_it \
#     # --num_fewshot 5 \
#     --device cuda:0 \
#     --batch_size 1 \
#     --limit 0.01 \
#     --seed 42 \
#     --output_path "/home/bruno/Documents/GitHub/ir-nlp-llama/evluation/plain_llamantino3_eval" \
#     --log_samples 