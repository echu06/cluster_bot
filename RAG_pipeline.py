import os
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# from langchain_core.vectorstores import VectorStoreRetriever
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import pickle

# Set OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pdf_path = "C:/Users/benmc/OneDrive/Wesleyan/QAC 387/Assignment 3/TeamProject387/ThematicAnalysis.pdf"
# pdf_loader = UnstructuredPDFLoader(pdf_path, mode="hi_res")
pdf_loader = PyMuPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

# check to see if it worked
print(len(pdf_docs))
print(pdf_docs[0].page_content)

# Combine both
# all_docs = pdf_docs + url_docs

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(pdf_docs)

# Create vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(pdf_docs, embedding=embedding_model)

# check to see if it worked
print(vectorstore.index.ntotal)

# Save vectorstore and metadata
vectorstore.save_local("vectorstore/faiss_index")
with open("vectorstore/doc_metadata.pkl", "wb") as f:
    pickle.dump(pdf_docs, f)

# View the raw vectors (numpy array)
# Get the FAISS index object
faiss_index = vectorstore.index

# How many vectors are stored?
print(f"Total vectors: {faiss_index.ntotal}")

# View the raw vectors (numpy array)
vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)

print(f"Shape of embedding matrix: {vectors.shape}")
print("Sample embedding (first vector):")
print(vectors[0])

# save embeddings to an numpy array
np.save("vectorstore/rag_embeddings.npy", vectors)


print("✅ Vector store with PDF successfully built and saved.")

# add reference section from documents to the vector store
ids = vectorstore.add_documents(documents=pdf_docs)

# Check pdf references
results = vectorstore.similarity_search("What are the vector ids?")
print(results[0])
