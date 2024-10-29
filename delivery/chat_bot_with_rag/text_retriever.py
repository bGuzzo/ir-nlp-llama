import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration params
# Chnage it
documents_folder_path = (
    "/home/bruno/Documents/GitHub/ir-nlp-llama/delivery/chat_bot_with_rag/documents"
)

# Load all txt files as documents
logger.info("Loading documents...")
loader = DirectoryLoader(documents_folder_path, glob="**/*.txt")
documents = loader.load()
logger.info("Documents loaded successfully")

# Split documents into chunks
logger.info("Split documents...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
logger.info("Documents splitted successfully")

logger.info("Initializing FAISS retriever...")

# Use only CPU here to avoid OOM error when running aside a language model
model_kwargs = {"device": "cpu"}
"""
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 maps sentences & paragraphs to a 384 dimensional 
dense vector space and can be used for tasks like clustering or semantic search.

HF Reference: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Paper Reference (Sentence-BERT): https://arxiv.org/abs/1908.10084
"""
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs=model_kwargs, 
    encode_kwargs = {'normalize_embeddings': True}
)


"""
Define a functiuon to convert embedding distance to [0, 1] similarity score.
HF Embedding return the distance, invert it to abtain a valid similarity measure. 
"""
def relevance_score_fn(distance: float) -> float:
    norm_score = 1.0 / (1.0 + abs(distance))
    logger.info(f"Distance: {distance}, Score: {norm_score}")
    return norm_score


"""
Verctor database using FAISS.
Faiss (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. 
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

Reference: https://github.com/facebookresearch/faiss
Paper: https://arxiv.org/abs/2401.08281
"""
db = FAISS.from_documents(docs, embeddings, relevance_score_fn=relevance_score_fn)

# Create retriever with 0.5 threshold to avoid capture useless information
retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": 0.5}
)

logger.info("FAISS Retriever ready")


def get_retrieved_text(user_prompt: str, max_words: int):
    """
    Retrieves relevant text from a FAISS vector database based on a user prompt.

    Args:
        user_prompt: The user's query or prompt.
        max_words: The maximum number of words to include in the retrieved text.
                If 0, RAG is skipped and an empty string is returned.

    Returns:
        A string containing the retrieved text, or an empty string if no relevant
        information is found or max_words is 0.
    """

    if max_words == 0:
        logger.warning(f"Skip RAG usage, max_words set to {max_words}")
        return ""

    logger.info(f"Retriever input query:\t{user_prompt}")
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_prompt)

    if len(retrieved_docs) == 0:
        logger.warning("No useful information found")
        return ""

    # Extract text content from retrieved documents
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

    if max_words > 0:
        # Text cutting to avoid OOM errors
        words = retrieved_text.split()
        retrieved_text = " ".join(words[:max_words])
        logger.warning(f"Cutting retrieved text to {max_words} words")

    logger.info(f"Retrieved text:\t{retrieved_text}")
    return retrieved_text

