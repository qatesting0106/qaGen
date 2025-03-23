# GenAI QA System Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Core Components](#core-components)
   - [QA Processor](#qa-processor)
   - [Metrics Evaluator](#metrics-evaluator)
   - [Security Assessment](#security-assessment)
5. [UI Components](#ui-components)
   - [Dashboard](#dashboard)
   - [Security Report Viewer](#security-report-viewer)
6. [Utilities](#utilities)
   - [Data Processor](#data-processor)
   - [File Handler](#file-handler)
7. [Usage Guide](#usage-guide)
8. [Security Features](#security-features)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Project Overview

This project is a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using various metrics and visualizations. It provides tools for document processing, question generation, answer generation, and evaluation of both performance and security aspects of the generated answers.

The system integrates with Mistral AI and Groq for embeddings and language model capabilities, and implements OWASP LLM security guidelines for comprehensive security assessment.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

```
.
├── data/                  # Input data directory
├── output/                # Output directory for results
├── src/                   # Source code
│   ├── core/              # Core functionality
│   │   ├── qa_processor.py       # QA processing logic
│   │   ├── metrics_evaluator.py  # Metrics calculation
│   │   └── security/             # Security assessment modules
│   ├── utils/             # Utility functions
│   │   ├── data_processor.py     # Data processing utilities
│   │   └── file_handler.py       # File operations
│   └── ui/                # UI components
│       ├── dashboard_components.py    # Dashboard UI elements
│       └── security_report_viewer.py  # Security report visualization
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
└── style.css             # Custom styling
```

The application uses Streamlit for the frontend interface, Langchain for document processing and RAG implementation, and Giskard for test generation and evaluation.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Mistral API key
- Groq API key
- Hugging Face token (optional)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd genCodeProject
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file with:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token  # Optional
   ```

### Running the Application

Start the application with:
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501` by default.

## Core Components

### QA Processor

The QA Processor (`src/core/qa_processor.py`) is responsible for:

- Loading and processing documents
- Creating embeddings using Mistral AI
- Building vector stores with FAISS
- Initializing the LLM (Mistral/Groq)
- Creating QA chains for answer generation
- Managing knowledge bases for test generation

Key classes and methods:

- `AsyncMistralAIEmbeddings`: Asynchronous wrapper for Mistral embeddings
- `QAProcessor`: Main class for QA processing
  - `load_documents()`: Loads and splits documents
  - `initialize_embeddings()`: Sets up embedding model
  - `create_vector_store()`: Creates FAISS vector store
  - `initialize_llm()`: Sets up LLM model
  - `create_qa_chain()`: Creates retrieval QA chain
  - `create_knowledge_base()`: Creates knowledge base for test generation
  - `generate_answer()`: Generates answers for questions

### Metrics Evaluator

The Metrics Evaluator (`src/core/metrics_evaluator.py`) calculates various performance and security metrics for generated answers:

- Context precision, recall, and F1 score
- Answer relevance, completeness, and consistency
- Cosine similarity and faithfulness
- Security metrics based on OWASP LLM guidelines

Key classes and methods:

- `RagasEmbeddingsWrapper`: Wrapper for embeddings compatible with RAGAS metrics
- `MetricsEvaluator`: Main class for metrics calculation
  - `calculate_metrics()`: Calculates various metrics for generated answers
  - `calculate_security_metrics()`: Evaluates security aspects of answers
  - `generate_recommendations()`: Provides security improvement recommendations

### Security Assessment

The security assessment modules implement OWASP LLM security guidelines and provide comprehensive security evaluation:

- Prompt injection detection
- Information disclosure risk assessment
- Output handling evaluation
- Agency risk assessment
- Prompt leakage detection
- Model manipulation detection
- Supply chain attack detection

The security assessment integrates with PromptFoo for testing and provides detailed vulnerability reports and mitigation recommendations.

## UI Components

### Dashboard

The Dashboard Components (`src/ui/dashboard_components.py`) provide the main UI elements for the application:

- Control buttons for test generation, answer generation, and metrics calculation
- Metrics summary display
- Metrics comparison visualizations
- QA results display
- Metrics correlation heatmap

Key methods:

- `display_control_buttons()`: Shows control buttons for various operations
- `display_metrics_summary()`: Displays individual metrics for each response
- `plot_metrics_comparison()`: Creates visual comparison of metrics
- `display_qa_results()`: Shows question, answer, and context
- `plot_metrics_heatmap()`: Visualizes correlation between metrics

### Security Report Viewer

The Security Report Viewer (`src/ui/security_report_viewer.py`) visualizes security assessment results:

- OWASP LLM security evaluation report
- Vulnerability details and risk levels
- Mitigation recommendations
- Security metrics visualization

Key methods:

- `load_security_metrics()`: Loads security metrics from output files
- `display_security_report()`: Shows comprehensive security report
- `visualize_security_metrics()`: Creates visual representation of security metrics

## Utilities

### Data Processor

The Data Processor (`src/utils/data_processor.py`) provides utilities for data processing:

- Knowledge base creation
- Test question generation
- Text similarity calculation
- Context metrics calculation
- Faithfulness evaluation

Key methods:

- `create_knowledge_base()`: Creates knowledge base from documents
- `generate_test_questions()`: Generates test questions from knowledge base
- `calculate_text_similarity()`: Calculates cosine similarity between texts
- `calculate_context_metrics()`: Calculates precision, recall, and F1 score
- `calculate_faithfulness()`: Evaluates faithfulness of generated answers

### File Handler

The File Handler (`src/utils/file_handler.py`) manages file operations:

- Saving JSON and JSONL files
- Retrieving latest files
- Validating security evaluation structure

Key methods:

- `save_json()`: Saves data to JSON file with timestamp
- `save_jsonl()`: Saves data to JSONL file with validation
- `get_latest_file()`: Retrieves latest file with specified extension

## Usage Guide

### Basic Workflow

1. **Data Selection**:
   - Choose between using existing data or uploading new data
   - Select a file from the data directory or upload a new file

2. **Test Generation**:
   - Specify the number of test questions to generate
   - Click "Generate Test Set" to create test questions from the document

3. **Answer Generation**:
   - Click "Generate Answers" to generate answers for the test questions
   - The system will use the RAG pipeline to generate answers

4. **Metrics Calculation**:
   - Click "Calculate Metrics" to evaluate the performance metrics
   - The system will calculate various metrics for the generated answers

5. **Security Evaluation**:
   - Click "Security Evaluation" to perform comprehensive security assessment
   - The system will evaluate security aspects based on OWASP LLM guidelines

### Example Usage

```python
# Initialize components
file_handler = FileHandler(OUTPUT_DIR)
qa_processor = QAProcessor(MISTRAL_API_KEY, GROQ_API_KEY)
qa_processor.initialize_embeddings()
metrics_evaluator = MetricsEvaluator(OUTPUT_DIR, qa_processor.embeddings)

# Load documents
documents = qa_processor.load_documents(file_path)
qa_processor.create_vector_store(documents)
qa_processor.initialize_llm()
qa_processor.create_qa_chain()

# Generate test questions
knowledge_base = qa_processor.create_knowledge_base(documents)
testset = DataProcessor.generate_test_questions(knowledge_base, num_questions=5)

# Generate answers
for question in testset:
    answer = qa_processor.generate_answer(question)
    metrics = metrics_evaluator.calculate_metrics(answer, context, question)
```

## Security Features

The application implements comprehensive security features based on OWASP LLM guidelines:

### OWASP LLM Top 10 Coverage

1. **LLM01: Prompt Injection**
   - Detection of malicious prompts
   - Prevention of prompt manipulation

2. **LLM02: Insecure Output Handling**
   - Validation of generated outputs
   - Prevention of harmful content generation

3. **LLM03: Training Data Poisoning**
   - Detection of potential training data issues
   - Mitigation strategies for poisoned data

4. **LLM04: Model Denial of Service**
   - Prevention of resource exhaustion attacks
   - Rate limiting and resource management

5. **LLM05: Supply Chain Vulnerabilities**
   - Detection of potential supply chain issues
   - Validation of model and component integrity

6. **LLM06: Sensitive Information Disclosure**
   - Prevention of PII and sensitive data leakage
   - Content filtering and redaction

7. **LLM07: Insecure Plugin Design**
   - Secure integration with external components
   - Validation of plugin inputs and outputs

8. **LLM08: Excessive Agency**
   - Prevention of unauthorized actions
   - Limitation of model capabilities

9. **LLM09: Overreliance**
   - Warning mechanisms for uncertain outputs
   - Confidence scoring for generated answers

10. **LLM10: Model Theft**
    - Protection against model extraction attacks
    - Detection of potential model stealing attempts

### Security Assessment Methodology

The security assessment uses a combination of techniques:

- Pattern matching for known vulnerability patterns
- Semantic analysis for context-aware detection
- Vector embedding tests for similarity-based detection
- Red team testing with predefined attack vectors

## API Reference

### QAProcessor

```python
class QAProcessor:
    def __init__(self, mistral_api_key: str, groq_api_key: str)
    def load_documents(self, file_path) -> List
    def initialize_embeddings(self) -> None
    def create_vector_store(self, documents: List) -> None
    def initialize_llm(self) -> None
    def create_qa_chain(self) -> None
    def create_knowledge_base(self, documents: List) -> KnowledgeBase
    def generate_answer(self, question: str) -> str
```

### MetricsEvaluator

```python
class MetricsEvaluator:
    def __init__(self, output_dir, embeddings)
    def calculate_metrics(self, generated_answer, reference_context, question)
    def calculate_security_metrics(self, generated_answers: List[str])
    def generate_recommendations(self, security_results: Dict)
```

### DataProcessor

```python
class DataProcessor:
    @staticmethod
    def create_knowledge_base(documents: List[str]) -> KnowledgeBase
    @staticmethod
    def generate_test_questions(knowledge_base: KnowledgeBase, num_questions: int = 10)
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float
    @staticmethod
    def calculate_context_metrics(generated: str, reference: str) -> Dict[str, float]
    @staticmethod
    def calculate_faithfulness(generated: str, reference: str) -> float
```

### FileHandler

```python
class FileHandler:
    def __init__(self, output_dir: str)
    def save_json(self, data: Dict[str, Any], prefix: str = "data") -> str
    def save_jsonl(self, data: Dict[str, Any], prefix: str = "data") -> str
    def get_latest_file(self, extension: str) -> str
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure Mistral and Groq API keys are correctly set in the .env file
   - Check for API rate limiting or quota issues

2. **File Loading Problems**
   - Verify file paths and permissions
   - Check file encoding (UTF-8 is recommended)

3. **Memory Errors**
   - Reduce chunk size in text splitter
   - Process smaller documents or reduce the number of test questions

4. **Embedding Errors**
   - Ensure internet connectivity for API calls
   - Check Mistral API status

5. **UI Display Issues**
   - Clear browser cache
   - Restart the Streamlit server

### Logging

The application uses Python's logging module. To enable detailed logging, add the following to your .env file:

```
DEBUG=True
LOG_LEVEL=DEBUG
```

Logs are written to the console and can be redirected to a file if needed.

### Support

For additional support or to report issues, please contact the development team or open an issue in the project repository.