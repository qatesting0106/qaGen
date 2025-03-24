# GenAI QA System Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Core Components](#core-components)
5. [UI Components](#ui-components)
6. [Advanced Usage](#advanced-usage)
7. [Security Guidelines](#security-guidelines)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Project Overview

This project is a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using various metrics and visualizations. It provides tools for document processing, question generation, answer generation, and evaluation of both performance and security aspects of the generated answers.

Key Features:
- Document processing and chunking with configurable parameters
- Advanced embedding generation using Mistral AI
- Efficient vector storage using FAISS
- Answer generation with Groq LLM
- Comprehensive metrics evaluation
- Security assessment based on OWASP LLM guidelines
- Interactive Streamlit dashboard
- Automated test generation using Giskard

## System Architecture

The application follows a modular architecture with clear separation of concerns:

```
/genCodeProject
├── data/                  # Input data and test cases
│   ├── raw/              # Raw input documents
│   └── processed/        # Processed and chunked documents
├── docs/                  # Project documentation
├── src/                   # Source code
│   ├── core/             # Core functionality
│   │   ├── qa_processor.py      # QA processing logic
│   │   ├── metrics_evaluator.py # Metrics calculation
│   │   └── security/            # Security assessment
│   │       ├── evaluator.py     # Main security evaluator
│   │       └── rules/           # Security rules
│   ├── ui/               # UI components
│   │   ├── components/   # Reusable UI components
│   │   └── pages/        # Dashboard pages
│   └── utils/            # Utility functions
│       ├── data_processor.py    # Data processing
│       ├── file_handler.py      # File operations
│       └── metrics_utils.py     # Metrics utilities
├── tests/                # Test suite
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
├── style.css            # Custom styling
└── main.py              # Application entry point
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
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
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MAX_CONTEXT_LENGTH=2000
   RISK_THRESHOLD=0.7
   ```

5. Initialize the application:
   ```bash
   python init_app.py  # Creates necessary directories and configs
   ```

## Core Components

### QA Processor

The QA Processor handles document processing, embedding generation, and answer generation:

1. **Document Processing**
   - Supports multiple file formats (txt, pdf, docx)
   - Configurable chunking parameters
   - Metadata extraction and preservation

2. **Embedding Generation**
   - Mistral AI integration for high-quality embeddings
   - Batch processing for efficiency
   - Caching mechanism for faster retrieval

3. **Answer Generation**
   - RAG pipeline with Groq LLM
   - Context-aware answer generation
   - Confidence score calculation

### Metrics Evaluator

Comprehensive evaluation system with multiple metric types:

1. **Context Metrics**
   - Precision: Accuracy of retrieved context
   - Recall: Completeness of context retrieval
   - F1 Score: Balanced measure of precision and recall

2. **Answer Quality Metrics**
   - Relevance: Answer alignment with question
   - Completeness: Coverage of question aspects
   - Consistency: Agreement with context

3. **Similarity Metrics**
   - Cosine Similarity: Vector space similarity
   - Faithfulness: Factual accuracy measurement

### Security Evaluator

Implements OWASP LLM security guidelines:

1. **Prompt Injection Detection**
   - Pattern-based detection
   - Behavioral analysis
   - Risk scoring

2. **Information Disclosure Prevention**
   - PII detection
   - Sensitive data filtering
   - Output sanitization

3. **Agency Risk Assessment**
   - Autonomy evaluation
   - Decision impact analysis
   - Control mechanism verification

## UI Components

### Dashboard

1. **Control Panel**
   - Document upload interface
   - Configuration settings
   - Process control buttons

2. **Visualization Components**
   - Interactive charts
   - Metric displays
   - Security assessment reports

3. **Results Display**
   - Answer presentation
   - Context highlighting
   - Source attribution

## Advanced Usage

### Custom Metrics

Implement custom metrics by extending the MetricsEvaluator class:

```python
class CustomMetricsEvaluator(MetricsEvaluator):
    def calculate_custom_metric(self, answer, context):
        # Implementation
        return score
```

### Security Rules

Add custom security rules:

```python
class CustomSecurityRule(SecurityRule):
    def evaluate(self, answer, prompt):
        # Rule implementation
        return risk_score
```

### Performance Optimization

1. **Embedding Optimization**
   - Batch size tuning
   - Caching strategy
   - Dimension reduction

2. **Vector Store Tuning**
   - Index optimization
   - Search parameter adjustment
   - Memory management

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce chunk size
   - Enable batch processing
   - Clear cache periodically

2. **Performance Issues**
   - Optimize vector store
   - Adjust batch sizes
   - Enable caching

3. **API Rate Limits**
   - Implement request throttling
   - Use batch processing
   - Cache results

### Error Messages

- `ValueError: Empty document list`: Ensure documents are properly loaded
- `AuthenticationError`: Check API keys in .env file
- `MemoryError`: Reduce chunk size or batch size
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

### Dashboard Components

The Dashboard Components (`src/ui/dashboard_components.py`) provide the main UI elements:

- Control buttons for test generation, answer generation, and metrics calculation
- Metrics visualization and comparison
- Security evaluation display
- QA results presentation

Key methods:

- `display_control_buttons()`: Shows control buttons for various operations
- `display_metrics_summary()`: Displays individual metrics for each response
- `plot_metrics_comparison()`: Creates visual comparison of metrics
- `plot_metrics_heatmap()`: Visualizes correlation between metrics
- `display_qa_results()`: Shows question, answer, and context

## Utilities

### Data Processor

The Data Processor (`src/utils/data_processor.py`) provides utilities for data processing:

- Document loading and preprocessing
- Test question generation
- Knowledge base creation
- Text similarity calculation
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