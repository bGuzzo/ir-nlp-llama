# Fine-Tuning and Evaluation of LLaMAntino 3 for Enhanced Italian Language Proficiency

This repository contains the complete artifacts of a research project conducted for the Natural Language Processing (NLP) course as part of a Master of Science degree. The project undertakes a comprehensive investigation into the fine-tuning of a state-of-the-art Large Language Model (LLM), `LLaMAntino-3-ANITA-8B-Inst-DPO-ITA`, on consumer-grade hardware.

## Abstract

This research delves into the fine-tuning of a large language model to enhance its proficiency in the Italian language. By employing a suite of optimization techniques—including 4-bit quantization, Low-Rank Adaptation (LoRA), and gradient checkpointing—we efficiently fine-tuned the model on consumer hardware using the Italian Wikipedia dataset. This methodology enabled the optimization of the model's performance while minimizing computational overhead. To assess the impact of our fine-tuning strategy, we conducted a comprehensive evaluation using a series of Italian benchmarks, including perplexity analysis, commonsense reasoning tasks (ARC-IT, Hellaswag-IT), and a custom-developed benchmark based on the Italian INVALSI scholastic tests. The project further explores the practical application of the fine-tuned model through the implementation of both standard and Retrieval-Augmented Generation (RAG) chatbot systems.

## Key Findings

The empirical results of this research yield several key insights into the effects of domain-specific fine-tuning on LLMs:

1.  **Improved Language Modeling on Domain-Specific Data:** The fine-tuned model, `Formal-LLaMAntino-3`, demonstrated a marked improvement in language modeling capabilities on the Italian Wikipedia dataset. It achieved a perplexity score of **11.39**, significantly outperforming both the base `LLaMAntino-3` model (**18.76**) and the original `Llama-3` model (**15.92**). This quantitatively validates the success of the fine-tuning process in adapting the model to the target domain.

2.  **Stylistic Specialization and Performance Trade-offs:** A qualitative analysis of the model's generative capabilities revealed a distinct stylistic shift towards a more formal, discursive, and scientific tone, consistent with the language used in Wikipedia articles. While this specialization is desirable for academic and formal contexts, it resulted in a performance degradation on general commonsense reasoning benchmarks such as ARC-IT and Hellaswag-IT. This suggests a trade-off between domain specialization and general-purpose reasoning capabilities.

3.  **Challenges in Task-Specific Adaptation:** A custom benchmark was developed using the Italian INVALSI scholastic tests to evaluate language comprehension in an educational context. All evaluated models, including the fine-tuned version, exhibited suboptimal performance on this benchmark. Our hypothesis is that the models, which are primarily optimized for generating contextually rich and verbose responses, struggle with tasks requiring concise, multiple-choice answers. This highlights the challenges in adapting generative models to specific, constrained-output tasks without further task-specific fine-tuning (e.g., reward training).

4.  **Efficacy of Retrieval-Augmented Generation (RAG):** The project successfully implemented a RAG system to augment the fine-tuned model with a domain-specific knowledge base on Italian literature. The resulting chatbot demonstrated a profound ability to answer specific questions by retrieving and synthesizing information from the provided documents, validating the efficacy of RAG in enhancing the factual grounding and domain expertise of LLMs.

For a more detailed analysis of these findings, please refer to the full academic report: [./llamantino-project/document.pdf](./llamantino-project/document.pdf)

## Repository Structure

The project is organized into a series of modules, each encapsulating a specific phase of the research workflow. The primary artifacts are located within the `delivery/` directory.

```
delivery/
├─── chat_bot/ -> [README](./delivery/chat_bot/README.md)
│    └─── (CLI and GUI chatbot implementations)
├─── chat_bot_with_rag/ -> [README](./delivery/chat_bot_with_rag/README.md)
│    └─── (RAG-enhanced GUI chatbot and knowledge base)
├─── evaluation/
│    └─── (Scripts for model evaluation on standard benchmarks)
├─── invalsi_benchmark/ -> [README](./delivery/invalsi_benchmark/README.md)
│    └─── (Custom INVALSI benchmark implementation and results)
├─── model_dump/
│    └─── Formal_LLaMAntino_3/ -> [Model Card](./delivery/model_dump/Formal_LLaMAntino_3/README.md)
│         └─── (Fine-tuned model adapters and configuration)
├─── perplexity/
│    └─── (Scripts and results for perplexity evaluation)
└─── training/
     └─── (Scripts for model fine-tuning)
```

## Frameworks and Technologies

This project leverages several key frameworks and libraries from the modern NLP ecosystem:

-   **[PyTorch](https://pytorch.org/):** The core deep learning framework.
-   **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index):** For model loading and text generation pipelines.
-   **[Hugging Face PEFT](https://huggingface.co/docs/peft/index):** For Parameter-Efficient Fine-Tuning, specifically LoRA.
-   **[Unsloth](https://github.com/unslothai/unsloth):** For optimized, memory-efficient training.
-   **[LangChain](https://www.langchain.com/):** For the implementation of the Retrieval-Augmented Generation (RAG) system.
-   **[FAISS](https://faiss.ai/):** For efficient similarity search in the RAG system's vector store.
