# Retrieval-Augmented Generation (RAG) Chatbot Module

This module presents an advanced implementation of a conversational agent that integrates a Retrieval-Augmented Generation (RAG) system. The primary objective is to enhance the capabilities of the fine-tuned `Formal-LLaMAntino-3` model by dynamically providing it with domain-specific information from a curated knowledge base. This approach allows the model to generate more accurate, detailed, and contextually relevant responses to user queries, particularly on topics related to Italian literature.

## Module Structure

The module is organized into the following key components:

*   `gui_chat_w_rag_formal_llamantino_3.py`: This script provides a comprehensive Graphical User Interface (GUI) for the RAG chatbot. Built with Tkinter, the GUI allows for seamless interaction with the model and offers real-time configuration of various parameters, including those for the language model (e.g., `temperature`, `top-p`) and the RAG system (e.g., `Max RAG Words`).

*   `text_retriever.py`: This script is the core of the RAG system. It is responsible for loading textual documents, segmenting them into manageable chunks, and generating vector embeddings using a `sentence-transformers` model. These embeddings are then indexed in a FAISS (Facebook AI Similarity Search) vector store, which enables efficient and rapid retrieval of relevant text based on semantic similarity to the user's query.

*   `documents/`: This directory serves as the knowledge base for the RAG system. It contains a collection of plain text files (`.txt`) on the subject of Italian literature, including documents on the *Divina Commedia*, the *Dolce Stil Novo*, and general Italian literature for secondary school.

## System Architecture

The RAG system operates through a multi-stage process:

1.  **User Input**: A user submits a query through the GUI.
2.  **Retrieval**: The `text_retriever.py` script processes the user's query by generating an embedding and performing a similarity search on the FAISS vector index. The most relevant text segments from the knowledge base are retrieved based on a predefined similarity threshold.
3.  **Augmentation**: The retrieved text is then prepended to the original user query, creating an augmented prompt that provides the model with rich, contextual information.
4.  **Generation**: This augmented prompt is subsequently passed to the `Formal-LLaMAntino-3` model, which leverages the provided context to generate a well-informed and accurate response.

## Usage

To launch the RAG chatbot, execute the following command in the terminal:

```bash
python gui_chat_w_rag_formal_llamantino_3.py
```
*Prerequisites: Ensure all dependencies listed in the main `requirements.txt` are installed.*

## Knowledge Base Customization

The knowledge base of this RAG system can be expanded by adding new plain text (`.txt`) files to the `documents/` directory. The `text_retriever.py` script will automatically load, process, and index any new documents upon initialization, thereby extending the chatbot's domain knowledge.
