from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.llms import HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.output_parsers import StrOutputParser

import datetime
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os
import json
import shutil


base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301"
model_name = "checkpoint-57301"

top_p = 0.95
temp = 0.6
n_token_out = 8
num_test_epoch = 2

print(f"Initialize on model: f{model_name}")
print(f"Model HF path: {base_model}")
print("="*shutil.get_terminal_size().columns)
print("Start loading model...")

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
# chat_template = tokenizer.chat_template

# print(f'Chat template {chat_template}')

print("Model loaded successfully")
print("="*shutil.get_terminal_size().columns)
print("\n")

sys = (
    # "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA. "
    "Rispondi in solo lingua italiana. "
    "Rispondi alle richieste degli utenti nel modo più conciso possibile, usando il minor numero di parole, senza sacrificare la chiarezza e la completezza dell'informazione. "
    "Privilegia risposte dirette e puntuali, evitando giri di parole e informazioni superflue. "
    "Utilizza elenchi puntati o numerati quando possibile per presentare informazioni in modo sintetico. "
    "Evita di ripetere informazioni già presenti nella domanda. "
    "Concentrati sull'essenziale, fornendo solo le informazioni strettamente necessarie per rispondere alla domanda. "
)

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,  # Adjust as needed
    temperature=0.6,  # Adjust as needed
    top_p=0.95,  # Adjust as needed
    # system_prompt=sys
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Load the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the documents and split them into chunks
books_path = "/home/bruno/Documents/GitHub/ir-nlp-llama/final_model/books"
texts = []
# Replace 'books' with the actual folder containing your books
for file in os.listdir(books_path):
    with open(os.path.join(books_path, file), "r") as f:
        texts.append(f.read())

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# Create the vectorstore
db = FAISS.from_documents(docs, embeddings)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=db.as_retriever())

# chat = [
#         {"role": "system", "content": sys},
#         {"role": "user", "content": "Spiega in breve il primo canto della divina commedia."}
#     ]

# prompt = f"""
#     System: {sys}
#     Richiesta: Spiega in breve il primo canto della divina commedia.
# """

# query = tokenizer.apply_chat_template(chat, tokenize=False)
# result = qa.invoke(query)['result']
prompt = f"""
            Contesto: {sys}
            Richiesta: Speiga in breve il primo canto della divina commedia.
        """
result = qa.invoke(prompt)
# print(result['result'])
result = result['result']
index = result.index("Helpful Answer: ")
clean_result = result[index + len("Helpful Answer: "):]
print(f"Result:\t{clean_result}")
# print('type: ', type(result['result']))
# print(f"Result:\t{result['result']}")
# for key, value in result.items() :
#     print('key', key)
# print('response type', type(result))
# print(result.content)