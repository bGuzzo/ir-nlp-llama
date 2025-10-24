"""
This script implements a text retrieval system for a Retrieval-Augmented Generation (RAG) chatbot.
It uses a FAISS vector database to efficiently search for and retrieve relevant text from a collection of documents
based on a user's query.

The script performs the following steps:
1.  Loads text documents from a specified directory.
2.  Splits the documents into smaller, manageable chunks.
3.  Uses a pre-trained Hugging Face sentence transformer model to create embeddings for the text chunks.
4.  Builds a FAISS vector database for efficient similarity search.
5.  Provides a function to retrieve relevant text based on a user prompt and a similarity threshold.

This implementation is designed to run on a CPU to avoid VRAM conflicts when used alongside a large language model.

References:
-   LangChain: https://python.langchain.com/docs/introduction
-   FAISS: https://github.com/facebookresearch/faiss
-   Sentence Transformers: https://www.sbert.net/
"""

import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Logger Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Configuration ---
# Path to the directory containing the text documents for the knowledge base.
documents_folder_path = "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/chat_bot_with_rag/documents"

# --- Document Loading and Preprocessing ---

logger.info("Loading documents...")
# Load all .txt files from the specified directory.
loader = DirectoryLoader(documents_folder_path, glob="**/*.txt")
documents = loader.load()
logger.info("Documents loaded successfully")

logger.info("Splitting documents...")
# Split the loaded documents into smaller chunks for more granular retrieval.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
logger.info("Documents split successfully")

# --- Retriever Initialization ---

logger.info("Initializing FAISS retriever...")

# Configure the embedding model to run on the CPU to conserve VRAM.
model_kwargs = {"device": "cpu"}

# Initialize the sentence transformer model for creating text embeddings.
# This model maps sentences and paragraphs to a 384-dimensional dense vector space.
# HF Reference: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# Paper Reference (Sentence-BERT): https://arxiv.org/abs/1908.10084
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs=model_kwargs,
    encode_kwargs={'normalize_embeddings': True}
)

def relevance_score_fn(distance: float) -> float:
    """Converts the embedding distance to a similarity score between 0 and 1."""
    norm_score = 1.0 / (1.0 + abs(distance))
    logger.info(f"Distance: {distance}, Score: {norm_score}")
    return norm_score

# Create a FAISS vector database from the document chunks and their embeddings.
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
# Reference: https://github.com/facebookresearch/faiss
# Paper: https://arxiv.org/abs/2401.08281
db = FAISS.from_documents(docs, embeddings, relevance_score_fn=relevance_score_fn)

# Create a retriever from the FAISS database.
# This retriever will find documents with a similarity score above the specified threshold.
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)

logger.info("FAISS Retriever ready")


def get_retrieved_text(user_prompt: str, max_words: int):
    """
    Retrieves relevant text from the FAISS vector database based on a user prompt.

    Args:
        user_prompt: The user's query.
        max_words: The maximum number of words to include in the retrieved text.
                   If 0, RAG is skipped.

    Returns:
        A string containing the retrieved text, or an empty string if no relevant
        information is found or if RAG is skipped.
    """
    if max_words == 0:
        logger.warning(f"Skipping RAG, max_words is set to {max_words}")
        return ""

    logger.info(f"Retriever input query: {user_prompt}")
    # Retrieve relevant documents from the vector database.
    retrieved_docs = retriever.invoke(user_prompt)

    if not retrieved_docs:
        logger.warning("No useful information found")
        return ""

    # Combine the content of the retrieved documents.
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

    # Truncate the retrieved text to the specified maximum number of words.
    if max_words > 0:
        words = retrieved_text.split()
        retrieved_text = " ".join(words[:max_words])
        logger.warning(f"Truncating retrieved text to {max_words} words")

    logger.info(f"Retrieved text: {retrieved_text}")
    return retrieved_text

