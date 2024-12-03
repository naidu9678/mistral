from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document  # Import Document class
from sentence_transformers import SentenceTransformer

# Initialize your model and embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your documents to be converted into embeddings
documents = [
    "Document 1 text content",
    "Document 2 text content",
    "Document 3 text content"
]

# Convert the string documents into LangChain Document objects
document_objects = [Document(page_content=doc) for doc in documents]

# Initialize HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store and save the index
faiss_index = FAISS.from_documents(document_objects, embedding_model)

# Optionally, save the FAISS index for future use
faiss_index.save_local("faiss_index/index.faiss")
print(f"FAISS index saved successfully.")
