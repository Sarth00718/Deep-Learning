# Practical 10: RAG - Retrieval Augmented Generation

## Overview
This practical demonstrates how to build a RAG (Retrieval Augmented Generation) system that can answer questions based on your documents using vector embeddings and large language models.

## What is RAG?
RAG combines:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Adding that information to the prompt
3. **Generation**: Using an LLM to generate answers based on retrieved context

## Files

### 1. `RAG.py`
Basic RAG implementation with:
- Document loading
- Text chunking
- Vector embeddings
- Similarity search
- LLM-based answer generation

### 2. `RAG_improved.py`
Enhanced version with:
- Better error handling
- Interactive Q&A loop
- Source document display
- Custom prompts
- Progress indicators

## Requirements

```bash
pip install langchain langchain-community langchain-groq langchain-huggingface
pip install chromadb sentence-transformers python-dotenv
```

## Setup

1. **Create `.env` file**:
```bash
cp .env.example .env
```

2. **Add your Groq API key** to `.env`:
```
GROQ_API_KEY=your_actual_api_key_here
```

3. **Add your documents** to the `Files/` directory

## Usage

### Basic Version
```bash
python RAG.py
```

### Improved Version (Recommended)
```bash
python RAG_improved.py
```

## How It Works

1. **Document Loading**: Loads text files from the `Files/` directory
2. **Text Splitting**: Breaks documents into manageable chunks (800 chars with 200 overlap)
3. **Embedding Creation**: Converts text chunks into vector embeddings using HuggingFace models
4. **Vector Storage**: Stores embeddings in ChromaDB for efficient retrieval
5. **Query Processing**: 
   - User asks a question
   - System finds most relevant chunks
   - LLM generates answer based on retrieved context

## Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Text Splitter│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Embeddings  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────┐
│  ChromaDB   │◄────┤  Query   │
└──────┬──────┘     └──────────┘
       │                  │
       │                  │
       ▼                  ▼
┌─────────────────────────┐
│    Retrieval + LLM      │
└───────────┬─────────────┘
            │
            ▼
      ┌──────────┐
      │  Answer  │
      └──────────┘
```

## Example Interaction

```
Your question: What is NLP?

Searching for relevant information...

======================================================================
ANSWER:
======================================================================
Natural Language Processing (NLP) is a branch of artificial intelligence 
that focuses on the interaction between computers and humans through 
natural language. It combines computational linguistics with statistical, 
machine learning, and deep learning models to enable computers to process 
and understand human language...

======================================================================
SOURCE DOCUMENTS:
======================================================================

Source 1:
Natural Language Processing (NLP) is a branch of artificial intelligence...

Source 2:
Key NLP Tasks include Text Classification, Named Entity Recognition...
```

## Key Components

### 1. Document Loader
- Loads text files
- Supports various formats (txt, pdf, docx with appropriate loaders)

### 2. Text Splitter
- `chunk_size=800`: Size of each text chunk
- `chunk_overlap=200`: Overlap between chunks for context preservation

### 3. Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Converts text to 384-dimensional vectors
- Fast and efficient for semantic search

### 4. Vector Store (ChromaDB)
- Persistent storage of embeddings
- Fast similarity search
- Supports filtering and metadata

### 5. LLM (Groq)
- Model: `llama-3.3-70b-versatile`
- Fast inference
- High-quality responses

## Customization

### Change Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for larger chunks
    chunk_overlap=100  # Adjust overlap
)
```

### Change Number of Retrieved Documents
```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 documents
)
```

### Change LLM Model
```python
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Different model
    temperature=0.5  # Adjust creativity
)
```

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution**: Create `.env` file with valid API key

### Issue: "File.txt not found"
**Solution**: Add text files to `Files/` directory

### Issue: ChromaDB errors
**Solution**: Delete `db/` directory and recreate:
```bash
rm -rf db/
python RAG_improved.py
```

### Issue: Slow embedding creation
**Solution**: First run takes time to download models. Subsequent runs are faster.

## Advanced Features

### Multiple Document Support
```python
loader = DirectoryLoader(
    'Files/',
    glob="**/*.txt",
    loader_cls=TextLoader
)
```

### PDF Support
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
```

### Metadata Filtering
```python
retriever = db.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "specific_doc.txt"}
    }
)
```

## Best Practices

1. **Document Preparation**: Clean and format documents before loading
2. **Chunk Size**: Balance between context and specificity (500-1000 chars)
3. **Overlap**: Use 10-20% of chunk size for overlap
4. **Number of Results**: Start with 3-5 retrieved documents
5. **Temperature**: Lower (0.1-0.3) for factual answers, higher (0.7-0.9) for creative responses

## Performance Tips

1. **Persistent Storage**: Vector store persists to disk, no need to recreate
2. **Batch Processing**: Process multiple queries efficiently
3. **Caching**: ChromaDB caches frequently accessed embeddings
4. **GPU**: Use GPU for faster embedding creation (if available)

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://console.groq.com/)
