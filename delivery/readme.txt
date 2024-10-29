Project folder structure 

.
├── chat_bot
│   ├── cli_chat_formal_llamantino_3.py : CLI chat-bot with fine-tuned model
│   ├── gui_chat_formal_llamantino_3.py : GUI chat-bot with fine-tuned model
│   └── readme.txt
├── chat_bot_with_rag
│   ├── documents : RAG document set
│   │   ├── divina_commedia.txt
│   │   ├── dolce_stil_novo.txt
│   │   └── letteratura_II_superiore.txt
│   ├── gui_chat_w_rag_formal_llamantino_3.py : GUI chat-bot with fine-tuned model and RAG
│   ├── readme.txt
│   └── text_retriever.py : RAG FAISS Retriever
├── evaluation
│   └── evaluate_lamantino_and_ours.sh : Command proposed by LLaMAntino authors to evaluate the model
├── invalsi_benchmark
│   ├── gemini_prompt.txt
│   ├── invalsi_test.py : Benchmark runner
│   ├── json_tests : INVALSI JSON dataset
│   │   ├── invalsi_ita_2007_2008.json
│   │   ├── invalsi_ita_2008_2009.json
│   │   └── invalsi_ita_2009_2010.json
│   ├── pdf_oringal : Original INVALSI PDF documents
│   │   ├── corrections
│   │   │   ├── griglia_correzione_invalsi_2010-2011_italiano_terza.pdf
│   │   │   ├── griglia_correzione_italiano_2007-2008_terza.pdf
│   │   │   ├── griglia_correzione_matematica_italiano_2009-2010_terza.pdf
│   │   │   └── griglie_correzione_matematica_italiano_2008-2009_terza.pdf
│   │   └── tests
│   │       ├── invalsi_italiano_2007-2008_terza.pdf
│   │       ├── invalsi_italiano_2008-2009_terza.pdf
│   │       ├── invalsi_italiano_2009-2010_terza.pdf
│   │       └── invalsi_italiano_2010-2011_terza.pdf
│   ├── readme.txt
│   └── results : JSON files with benchmark results
│       ├── LLaMA-3-8B
│       │   ├── INVALSI_Llama-3_shots_0_top-p_0.95_temp_0.6_token_4_2024-10-26_14-57-30.json
│       │   ├── INVALSI_Llama-3_shots_1_top-p_0.95_temp_0.6_token_4_2024-10-26_15-02-19.json
│       │   └── INVALSI_Llama-3_shots_2_top-p_0.95_temp_0.6_token_4_2024-10-26_15-08-04.json
│       ├── LLaMAntino-3-8B
│       │   ├── INVALSI_LLaMAntino-3_shots_0_top-p_0.95_temp_0.6_token_4_2024-10-26_14-37-23.json
│       │   ├── INVALSI_LLaMAntino-3_shots_1_top-p_0.95_temp_0.6_token_4_2024-10-26_14-42-55.json
│       │   └── INVALSI_LLaMAntino-3_shots_2_top-p_0.95_temp_0.6_token_4_2024-10-26_14-48-11.json
│       └── Wiki-LLaMAntino-3-8B
│           ├── INVALSI_Wiki-LLaMAntino_shots_0_top-p_0.95_temp_0.6_token_4_2024-10-26_14-19-34.json
│           ├── INVALSI_Wiki-LLaMAntino_shots_1_top-p_0.95_temp_0.6_token_4_2024-10-26_14-25-34.json
│           └── INVALSI_Wiki-LLaMAntino_shots_2_top-p_0.95_temp_0.6_token_4_2024-10-26_14-32-10.json
├── model_dump
│   ├── Formal_LLaMAntino_3 : fine-tuned model dump foder
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── optimizer.pt
│   │   ├── README.md
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── trainer_state.json
│   │   └── training_args.bin
│   └── Formal_LLaMAntino_3.zip : fine-tuned model dump foder zipped
├── perplexity
│   ├── PPL_Wiki_IT_all_2024-10-24_09-27-44.json : perplexity evaluation results
│   └── wiki_perplexity.py : perplexity evaluation script
├── requirements.txt : Pyhton requirements file
└── training
    ├── llamantino_finetune_original.py : original fine-tuning script proposed by Llamantino authors
    └── llamantino_wiki_train.py : The used py script to fine-tune the model on Wikipedia Dataset