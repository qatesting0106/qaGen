# RAG System Evaluation Framework

This project provides a framework for evaluating RAG (Retrieval-Augmented Generation) systems using various metrics and visualizations.

## Project Structure

```
.
├── data/                  # Input data directory
├── output/               # Output directory for results
├── src/                  # Source code
│   ├── core/            # Core functionality
│   ├── utils/           # Utility functions
│   └── ui/              # UI components
├── .env                 # Environment variables
├── requirements.txt     # Project dependencies
└── style.css           # Custom styling
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

Run the application:
```bash
streamlit run src/main.py
```

## Features

- Document processing and chunking
- Question generation from documents
- Answer generation using RAG
- Comprehensive metrics evaluation
- Interactive dashboard visualization
- Results export in JSON and HTML formats

## Metrics

- Cosine Similarity
- Context Precision
- Context Recall
- Context F1 Score
- Faithfulness