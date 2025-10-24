# Project Delivery: Fine-Tuning and Evaluation of LLaMAntino 3 for Enhanced Italian Language Proficiency

## Abstract

This project presents a comprehensive workflow for the fine-tuning, evaluation, and application of the `LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` model, with the primary objective of enhancing its proficiency and specialization in the Italian language. Leveraging advanced optimization techniques, including 4-bit quantization and Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), the fine-tuning process was successfully executed on consumer-grade hardware. The performance of the resultant model, `Formal-LLaMAntino-3`, was rigorously assessed through a series of quantitative benchmarks and qualitative analyses. This document provides a detailed overview of the project's structure, methodology, and key outcomes.

## Workflow and Methodology

The project was structured around a systematic workflow encompassing model fine-tuning, multi-faceted evaluation, and practical application development.

1.  **Model Fine-Tuning:**
    The foundational `LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` model was subjected to a supervised fine-tuning (SFT) process on the complete Italian Wikipedia dataset (`wikimedia/wikipedia`, `20231101.it` split). This was conducted to augment the model's knowledge base and to instill a more formal, scientific style of discourse. The training was made feasible on an 8GB GPU through the synergistic application of:
    *   **4-bit Quantization:** To reduce the model's memory footprint.
    *   **Low-Rank Adaptation (LoRA):** To minimize the number of trainable parameters.
    *   **Gradient Checkpointing:** To further conserve memory during training.
    The training script `delivery/training/llamantino_wiki_train.py` orchestrates this process.

2.  **Model Evaluation:**
    A multi-pronged evaluation strategy was adopted to holistically assess the performance of the fine-tuned model in comparison to its base model and the original Llama 3 model.
    *   **Perplexity Analysis:** The perplexity of the models was computed on a held-out subset of the Italian Wikipedia dataset to quantify improvements in language modeling capabilities. This is implemented in `delivery/perplexity/wiki_perplexity.py`.
    *   **Standardized Benchmarking:** The `lm-evaluation-harness` framework was employed to evaluate the models on Italian versions of established benchmarks, namely ARC (AI2 Reasoning Challenge) and HellaSwag, to assess commonsense reasoning.
    *   **INVALSI Benchmark:** A novel benchmark was developed using official Italian INVALSI tests for middle school students. This benchmark, located in the `delivery/invalsi_benchmark/` directory, is designed to provide a granular assessment of the models' Italian language comprehension in a formal educational context. The evaluation is based on the ROUGE-1 precision score.

3.  **Application Development:**
    To demonstrate the practical utility of the fine-tuned model, three distinct chatbot applications were developed:
    *   A Command-Line Interface (CLI) chatbot.
    *   A Graphical User Interface (GUI) chatbot with configurable parameters.
    *   A GUI chatbot enhanced with a Retrieval-Augmented Generation (RAG) system, leveraging a knowledge base of Italian literature to provide domain-specific, context-aware responses.

## Repository Structure

The `delivery/` directory is organized as follows, encapsulating all artifacts of this research project:

```
delivery/
├─── chat_bot/
│    ├─── cli_chat_formal_llamantino_3.py
│    ├─── gui_chat_formal_llamantino_3.py
│    └─── README.md
├─── chat_bot_with_rag/
│    ├─── gui_chat_w_rag_formal_llamantino_3.py
│    ├─── text_retriever.py
│    ├─── documents/
│    │    ├─── divina_commedia.txt
│    │    ├─── dolce_stil_novo.txt
│    │    └─── letteratura_II_superiore.txt
│    └─── README.md
├─── evaluation/
│    └─── evaluate_lamantino_and_ours.sh
├─── invalsi_benchmark/
│    ├─── invalsi_test.py
│    ├─── gemini_prompt.txt
│    ├─── json_tests/
│    │    ├─── invalsi_ita_2007_2008.json
│    │    ├─── ...
│    ├─── pdf_oringal/
│    │    ├─── ...
│    ├─── results/
│    │    ├─── ...
│    └─── README.md
├─── model_dump/
│    └─── Formal_LLaMAntino_3/
│         ├─── adapter_config.json
│         ├─── README.md
│         └─── ...
├─── perplexity/
│    ├─── wiki_perplexity.py
│    └─── PPL_Wiki_IT_all_2024-10-24_09-27-44.json
└─── training/
     ├─── llamantino_finetune_original.py
     └─── llamantino_wiki_train.py
```

## Conclusion

This project successfully demonstrates the feasibility of fine-tuning a large language model on consumer-grade hardware through the application of state-of-the-art optimization techniques. The comprehensive evaluation framework, including the novel INVALSI benchmark, provides valuable insights into the model's performance and the trade-offs associated with domain-specific fine-tuning. The developed chatbot applications serve as tangible proof-of-concepts for the practical deployment of the fine-tuned model.
