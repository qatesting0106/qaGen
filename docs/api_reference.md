# GenAI QA System API Reference

## Overview

This API reference provides detailed information about the classes and methods available in the GenAI QA System. It is intended for developers who want to use the system programmatically or extend its functionality.

## Core Components

### QAProcessor

The `QAProcessor` class handles document loading, embedding creation, vector store building, and answer generation.

```python
class QAProcessor:
    def __init__(self, mistral_api_key: str, groq_api_key: str)
```

**Parameters:**
- `mistral_api_key` (str): API key for Mistral AI embeddings
- `groq_api_key` (str): API key for Groq language model

#### Methods

```python
def load_documents(self, file_path) -> List
```

**Description:** Load and split documents from the given file path or uploaded file.

**Parameters:**
- `file_path` (str or UploadedFile): Path to the document file or Streamlit UploadedFile object

**Returns:**
- List of document chunks

**Raises:**
- `FileNotFoundError`: If the file does not exist
- `Exception`: If there is an error loading the document

---

```python
def initialize_embeddings(self) -> None
```

**Description:** Initialize the Mistral AI embeddings model.

**Returns:** None

---

```python
def create_vector_store(self, documents: List) -> None
```

**Description:** Create FAISS vector store from documents.

**Parameters:**
- `documents` (List): List of document chunks

**Returns:** None

---

```python
def initialize_llm(self) -> None
```

**Description:** Initialize the LLM model using GROQ API.

**Returns:** None

---

```python
def create_qa_chain(self) -> None
```

**Description:** Create retrieval QA chain for question answering.

**Returns:** None

---

```python
def create_knowledge_base(self, documents: List) -> KnowledgeBase
```

**Description:** Create a