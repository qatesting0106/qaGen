# GenAI QA System Quick Reference Guide

## Overview

This system is a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using various metrics and visualizations. It integrates with Mistral AI and Groq for embeddings and language model capabilities, and implements OWASP LLM security guidelines.

## Key Components

### Core Components

- **QA Processor**: Handles document loading, embedding creation, vector store building, and answer generation
- **Metrics Evaluator**: Calculates performance and security metrics for generated answers
- **Security Assessment**: Implements OWASP LLM security guidelines for comprehensive evaluation

### UI Components

- **Dashboard**: Provides control buttons, metrics visualization, and QA results display
- **Security Report Viewer**: Visualizes security assessment results and recommendations

### Utilities

- **Data Processor**: Handles knowledge base creation, test generation, and similarity calculations
- **File Handler**: Manages file operations and security validation

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Create a `.env` file with:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Run Application**:
   ```bash
   streamlit run main.py
   ```

## Basic Workflow

1. **Select Data**: Choose existing data or upload new data
2. **Generate Test Set**: Create test questions from the document
3. **Generate Answers**: Generate answers using the RAG pipeline
4. **Calculate Metrics**: Evaluate performance metrics
5. **Security Evaluation**: Perform security assessment

## Security Features

Implements OWASP LLM Top 10 security guidelines:
- Prompt Injection Detection
- Insecure Output Handling
- Training Data Poisoning Detection
- Model Denial of Service Prevention
- Supply Chain Vulnerability Detection
- Sensitive Information Disclosure Prevention
- Insecure Plugin Design Detection
- Excessive Agency Prevention
- Overreliance Warning
- Model Theft Protection

## Key Metrics

- **Context Metrics**: Precision, Recall, F1 Score
- **Answer Quality**: Relevance, Completeness, Consistency
- **Similarity Metrics**: Cosine Similarity, Faithfulness
- **Security Metrics**: Various risk assessments based on OWASP guidelines

## Troubleshooting

- **API Key Issues**: Verify Mistral and Groq API keys in .env file
- **File Loading Problems**: Check file paths and encoding (UTF-8 recommended)
- **Memory Errors**: Reduce chunk size or process smaller documents
- **Embedding Errors**: Ensure internet connectivity for API calls
- **UI Display Issues**: Clear browser cache or restart Streamlit server

For detailed documentation, refer to the [Project Documentation](./project_documentation.md).