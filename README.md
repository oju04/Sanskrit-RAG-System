# Sanskrit-RAG-System
This project implements a complete RAG (Retrieval-Augmented Generation) system for Sanskrit documents, optimized for CPU-based inference. 
Overview
Here is your rewritten, emoji-free, fully humanized version.
Tone: clear, professional, natural, and human-like.
Formatting preserved.

---

# Sanskrit Document RAG System

**Author:** Ojas Kamde
**Project:** Retrieval-Augmented Generation for Sanskrit Documents
**Optimization:** CPU-only inference

---

## Overview

This project provides a complete RAG (Retrieval-Augmented Generation) system designed specifically for Sanskrit documents, optimized to run smoothly on CPU-based environments. The system can:

* Ingest Sanskrit documents (txt, pdf, docx)
* Preprocess and clean Sanskrit text
* Split content into meaningful chunks
* Retrieve relevant information using TF-IDF keyword-based search
* Generate contextual responses using rule-based logic
* Accept queries in both English and Sanskrit

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Sanskrit RAG System                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Document   │───▶│ Preprocessor │───▶│  Chunker  │ │
│  │    Loader    │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └─────┬─────┘ │
│                                                  │       │
│                                                  ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Response   │◀───│   Retriever  │◀───│  Index    │ │
│  │  Generator   │    │  (TF-IDF)    │    │  Builder  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│                                                           │
│         ▲                                                 │
│         │                                                 │
│    User Query                                             │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Document Loader:** Reads Sanskrit documents from various formats
2. **Preprocessor:** Normalizes and cleans text
3. **Chunker:** Splits large text into structured chunks
4. **Index Builder:** Creates TF-IDF vector index
5. **Retriever:** Finds the most relevant chunks
6. **Generator:** Produces rule-based responses

---

## Installation

### Prerequisites

* Python 3.8 or higher
* pip
* At least 2GB RAM (4GB recommended)

### Setup Steps

1. **Navigate to the project folder:**

```bash
cd RAG_Sanskrit_Ojas_Kamde
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation:**

```bash
python -c "import numpy, sklearn; print('Setup successful!')"
```

---

## Usage

### Basic Usage

1. **Prepare your documents**

   * Place all Sanskrit text files in the `data/` folder
   * Accepted types: `.txt`, `.pdf`, `.docx`

2. **Run the system**

```bash
python code/main.py
```

3. During execution, the system will:

   * Load and index documents
   * Run sample queries
   * Enter interactive mode

### Example Queries

```python
# English
"Tell me about the foolish servant."
"What moral lesson is taught about divine help?"
"How did Kalidasa show his cleverness?"

# Sanskrit
"कालीदासस्य कथा वदतु"
"मूर्खभृत्यस्य विषये किं कथितम्?"
```

### Programmatic Usage

```python
from main import SanskritRAGSystem

rag = SanskritRAGSystem()

rag.ingest_document("data/Rag-docs.txt")

rag.save_model("models")

result = rag.query("What is the moral of these stories?")
print(result['answer'])
print("Confidence:", result['confidence'])
```

---

## Project Structure

```
RAG_Sanskrit_Ojas_Kamde/
├── code/
│   ├── main.py
│   ├── document_loader.py
│   ├── preprocessor.py
│   ├── retriever.py
│   └── generator.py
├── data/
│   ├── Rag-docs.txt
│   └── README.md
├── models/
│   └── index.pkl
├── report/
│   └── Technical_Report.pdf
├── requirements.txt
└── README.md
```

---

## Configuration

### Changing Chunk Size

```python
preprocessor = SanskritPreprocessor()
chunks = preprocessor.create_chunks(story, chunk_size=300)
```

### Changing Number of Retrieved Results

```python
result = rag.query("Your question", top_k=5)
```

### Custom Model Paths

```python
rag.save_model("custom/models")
rag.load_model("custom/models")
```

---

## Performance Metrics

### System Specifications

* CPU usage: 50–70% on a single core
* Memory usage: approx 500MB for 50 chunks
* Indexing time: 2–3 seconds for 1000 lines
* Retrieval time: 50–100 ms
* Generation time: 10–20 ms

### Retrieval Accuracy

* Precision@3: ~85%
* Recall@3: ~75%
* F1 Score: ~80%

---

## Features

### Completed

* Multi-format document support
* Sanskrit-aware preprocessing
* TF-IDF retrieval
* Cosine similarity ranking
* Rule-based response generation
* Bilingual query handling
* Model saving and loading
* Interactive mode
* Confidence scoring

### Planned Enhancements

* Transformer-based embeddings
* Advanced Sanskrit NLP
* Web interface with Streamlit/Flask
* Multi-document comparison
* Export to PDF 

---

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'numpy'`
**Fix:** Run `pip install -r requirements.txt`

**Problem:** `FileNotFoundError: Rag-docs.txt missing`
**Fix:** Ensure the file exists in the correct directory

**Problem:** Retrieval feels inaccurate
**Fix:**

* Increase chunk size
* Adjust IDF scoring settings
* Add more documents

**Problem:** High memory consumption
**Fix:**

* Reduce chunk size
* Use batch processing
* Prefer sparse matrix formats

---

## Technical Details

### TF-IDF Formula

```
TF(t,d) = count(t in d) / total words in d
IDF(t) = log((N + 1) / (df(t) + 1)) + 1
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### Cosine Similarity

```
similarity(q,d) = (q · d) / (||q|| × ||d||)
```

### Chunking Strategy

* Sentence-level splitting using Devanagari punctuation
* Default max chunk size: 200 words
* No overlap
* Story titles retained for reference

---

## Contributing

To extend or improve the system:

1. Add retrieval methods in `retriever.py`
2. Enhance response generation logic in `generator.py`
3. Introduce new formats in `document_loader.py`
4. Refine preprocessing rules in `preprocessor.py`

---

