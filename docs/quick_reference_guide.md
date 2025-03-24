# GenAI QA System Quick Reference Guide

## Overview

This system is a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using various metrics and visualizations. It integrates with Mistral AI for embeddings and Groq for language model capabilities, featuring a modern Streamlit-based dashboard for visualizing metrics and implementing OWASP LLM security guidelines.

## Key Components

### Core Components

- **QA Processor**: 
  - Document loading and preprocessing
  - Embedding creation using Mistral AI
  - FAISS vector store building
  - Answer generation using Groq LLM
  - Context retrieval and ranking

- **Metrics Evaluator**: 
  - Context precision, recall, and F1 score calculation
  - Answer quality assessment (relevance, completeness, consistency)
  - Similarity metrics computation (cosine similarity, faithfulness)
  - Performance metrics visualization

- **Security Evaluator**: 
  - OWASP LLM security guidelines implementation
  - Prompt injection detection
  - Information disclosure risk assessment
  - Output validation and sanitization
  - Agency risk evaluation

### UI Components

- **Dashboard Components**: 
  - Interactive control panel for document upload and processing
  - Real-time metrics visualization with charts and graphs
  - Comprehensive security evaluation display
  - QA results presentation with context highlighting
  - Export functionality for results and reports

### Utilities

- **Data Processor**: 
  - Document loading and format validation
  - Test case generation using Giskard
  - Knowledge base creation and management
  - Data preprocessing and cleaning

- **File Handler**: 
  - File operations and validation
  - Format conversion and normalization
  - Error handling and logging
  - Output file management

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

## Metrics

- **Context Metrics**: Precision, Recall, F1 Score
- **Answer Quality**: Relevance, Completeness, Consistency
- **Similarity Metrics**: Cosine Similarity, Faithfulness
- **Security Metrics**: Prompt injection risk, Information disclosure risk, Output handling risk, Agency risk

## Basic Usage

1. **Start Application**:
   ```bash
   streamlit run main.py
   ```

2. **Configure API Keys**:
   Create a `.env` file with:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Basic Workflow**:
   - Select or upload document
   - Generate test questions
   - Generate answers using RAG pipeline
   - View metrics and visualizations
   - Evaluate security aspects

For detailed documentation, refer to the [Project Documentation](./project_documentation.md).