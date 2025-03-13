# LLaMAntino 3: Further Fine-Tuning on the Italian Language

## Overview

This project explores the fine-tuning of the LLaMAntino 3 large language model (LLM) to enhance its Italian language capabilities. We employ optimization techniques, including 4-bit quantization, Low-Rank Adaptation (LoRA), and gradient checkpointing, to enable efficient fine-tuning on consumer hardware. 
## Project Structure

The repository is organized as follows:

* `chat_bot`: Contains the CLI and GUI chatbot implementations using the fine-tuned model.
* `chat_bot_with_rag`: Includes the GUI chatbot with RAG implementation, along with document sets and the text retriever.
* `evaluation`: Provides scripts for evaluating the model on various benchmarks.
* `invalsi_benchmark`: Contains the code and data for the INVALSI benchmark, used to test the model's Italian language comprehension.
* `model_dump`: Stores the fine-tuned model dumps.
* `perplexity`: Includes scripts and results for perplexity evaluation.
* `training`: Contains the original fine-tuning script and the script used for fine-tuning on the Wikipedia dataset.
* `delivery`: This directory contains all the folders mentioned above, scripts, a requirements.txt file, and a readme.txt file.
* `requirements.txt`: Lists the Python dependencies for the project.
* `Relazione_IR-NLP_Bruno_Guzzo.pdf`: A report detailing the project. 

# LLaMAntino 3: Further Fine-Tuning on the Italian Language

## Project Description

This project focuses on fine-tuning a large language model (LLM) for enhanced Italian language proficiency. We employ 4-bit quantization, gradient checkpointing, and low-rank adaptation to efficiently fine-tune the model on consumer hardware. The project explores the potential of incorporating Retrieval-Augmented Generation (RAG) with a chatbot approach.

## Key Components

* **Base Model**: Meta LLaMA and LLaMA 3.
* **Fine-tuning**: LLaMAntino 3 8B, an Italian language LLM based on Meta's LLaMA 3. 
* **Optimization**: 4-bit quantization, Low-Rank Adaptation (LoRA), and gradient checkpointing. 
* **Fine-tuning Dataset**: Italian Wikipedia dataset. 
* **Evaluation**: Perplexity score and common benchmark scores (ARC-it, Hellaswag-it).
* **Chatbot Implementation**: Includes chat template, behavioral instructions, and empirical results. 
* **RAG Implementation**: Retrieval-Augmented Generation to enhance chatbot capabilities. 
* **INVALSI Benchmark**: Evaluation using Italian language comprehension tests. 

## Results

* The fine-tuned model outperformed the original model on a perplexity test.
* The fine-tuned model showed a bias towards a more formal and scientific language style.
* The fine-tuned model did not perform well on the INVALSI benchmark.
* The RAG implementation demonstrated strong context understanding and information retrieval.

## Dependencies

* Python 3.10
* PyTorch 2.4.1
* Transformers 4.45.2
* Datasets 2.0.1
* Langchain 0.3.4
* Gradio 4.17.1
* FAISS 1.9.0
* sentence-transformers 2.2.0
* bitsandbytes 0.41.1
* Unsloth 0.9.0
* Jinja2
* A complete list of dependencies can be found in `requirements.txt`.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/username/your-repo.git](https://www.google.com/search?q=https://github.com/username/your-repo.git)
    ```

2.  Navigate to the project directory:

    ```bash
    cd your-repo
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

###   Chatbot

1.  To run the command-line interface (CLI) chatbot:

    ```bash
    python delivery/chat_bot/cli_chat_formal_llamantino_3.py
    ```

2.  To run the graphical user interface (GUI) chatbot:

    ```bash
    python delivery/chat_bot/gui_chat_formal_llamantino_3.py
    ```

###   Chatbot with RAG

1.  To run the GUI chatbot with RAG:

    ```bash
    python delivery/chat_bot_with_rag/gui_chat_w_rag_formal_llamantino_3.py
    ```

###   Evaluation

1.  To evaluate the model using the original LLaMAntino 3 evaluation script:

    ```bash
    bash delivery/evaluation/evaluate_lamantino_and_ours.sh
    ```

###   INVALSI Benchmark

1.  To run the INVALSI benchmark:

    ```bash
    python delivery/invalsi_benchmark/invalsi_test.py
    ```

## Model Fine-tuning

The script used for fine-tuning the model on the Wikipedia dataset is located at `delivery/training/llamantino_wiki_train.py`.

## Datasets

* Italian Wikipedia dataset: Used for fine-tuning.
* INVALSI dataset: Used for evaluation.
* RAG dataset: A custom dataset of Italian literature documents