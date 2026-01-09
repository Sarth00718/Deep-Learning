# =============================================================
# RAG - Retrieval Augmented Generation
# =============================================================

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os

# -------------------------------------------------------------
# File Paths Setup
# -------------------------------------------------------------
cur_dir = os.path.dirname(__file__)
file_path = os.path.join(cur_dir, "Files", "File.txt")
persist_dir = os.path.join(cur_dir, "db", "chroma_db")

if not os.path.exists(persist_dir):
    print("DB directory not found, creating...")
    os.makedirs(persist_dir)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found")

# -------------------------------------------------------------
# Load the document
# -------------------------------------------------------------
loader = TextLoader(file_path=file_path)
docs = loader.load()

print("Document loaded successfully!")
print("Type:", type(docs))
print("Content Sample:\n", docs[0].page_content[:1000])

# -------------------------------------------------------------
# Split text into chunks
# -------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# -------------------------------------------------------------
# Create embeddings
# -------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------------------------------------
# Create or load vector store
# -------------------------------------------------------------
if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    print("New Chroma vector store created and persisted.")
else:
    print("Loading existing Chroma vector store...")
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

# -------------------------------------------------------------
# Retrieve similar documents
# -------------------------------------------------------------
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

query = input("Enter your query: ")

retrieved_docs = retriever.invoke(query)

print("\n=== Retrieved Documents ===\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)
    print("Source:", doc.metadata)

# -------------------------------------------------------------
# Setup ChatGroq model
# -------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# -------------------------------------------------------------
# Combine context and generate final answer
# -------------------------------------------------------------
context = "\n".join([doc.page_content for doc in retrieved_docs])

response = model.invoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
])

print("\n=== Final Answer ===\n")
print(response)