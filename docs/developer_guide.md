# GenAI QA System Developer Guide

## Introduction

This developer guide provides information for developers who want to extend or modify the GenAI QA System. It covers the system's architecture, component interactions, and guidelines for adding new features.

## System Architecture

The system follows a modular architecture with the following components:

1. **Core Components**
   - QA Processor: Handles document processing and answer generation
   - Metrics Evaluator: Calculates performance and security metrics
   - Security Assessment: Implements OWASP LLM security guidelines

2. **UI Components**
   - Dashboard: Provides the main user interface
   - Security Report Viewer: Visualizes security assessment results

3. **Utilities**
   - Data Processor: Handles data processing and test generation
   - File Handler: Manages file operations and validation

## Component Interactions

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │────▶│  Dashboard  │────▶│ QA Processor │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Security   │◀────│   Metrics   │
                    │   Report    │     │  Evaluator  │
                    └─────────────┘     └─────────────┘
                          ▲                    ▲
                          │                    │
                    ┌─────────────┐     ┌─────────────┐
                    │    Data     │     │    File     │
                    │  Processor  │────▶│   Handler   │
                    └─────────────┘     └─────────────┘
```

## Adding New Features

### Adding a New Metric

1. **Update Metrics Evaluator**
   - Add a new method to `MetricsEvaluator` class in `src/core/metrics_evaluator.py`
   - Implement the metric calculation logic
   - Update the `calculate_metrics()` method to include the new metric

```python
def calculate_new_metric(self, generated_answer, reference_context, question):
    # Implement metric calculation logic
    return new_metric_value

def calculate_metrics(self, generated_answer, reference_context, question):
    # Existing code...
    
    # Add new metric
    metrics['new_metric'] = self.calculate_new_metric(generated_answer, reference_context, question)
    
    # Rest of the method...
```

2. **Update Dashboard Components**
   - Add the new metric to the appropriate metric group in `src/ui/dashboard_components.py`
   - Update visualization components if needed

```python
metric_groups = {
    # Existing groups...
    'New Metric Group': ['new_metric', 'other_related_metric'],
    # Or add to existing group
    'Similarity Metrics': ['cosine_similarity', 'faithfulness', 'new_metric'],
}
```

### Adding a New Security Test

1. **Create Test Method**
   - Add a new test method to the security assessment module
   - Implement detection logic for the new vulnerability type

```python
def _check_new_vulnerability(self, generated_answer):
    # Implement vulnerability detection logic
    vulnerability_count = 0
    # Detection code...
    return vulnerability_count
```

2. **Update Security Results**
   - Add the new test to the security results dictionary
   - Define risk level and test name

```python
security_results.update({
    'LLM11': {
        'vulnerability_count': self._check_new_vulnerability(generated_answer),
        'risk_level': self._determine_risk_level('LLM11'),
        'detected_payloads': [],
        'test_name': 'New Vulnerability Test',
        'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
})
```

3. **Update Security Report Viewer**
   - Add the new vulnerability type to the security report visualization

### Adding a New UI Component

1. **Create Component Class**
   - Create a new class in the `src/ui` directory
   - Implement the component's functionality

```python
class NewComponent:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def display_component(self, data):
        # Implement component display logic
        st.header("New Component")
        # Display code...
```

2. **Integrate with Main Application**
   - Import and initialize the new component in `main.py`
   - Add the component to the UI flow

## Working with External APIs

### Mistral AI Integration

The system uses Mistral AI for embeddings. To modify or extend this integration:

1. Update the `AsyncMistralAIEmbeddings` class in `src/core/qa_processor.py`
2. Ensure compatibility with the existing embedding interface
3. Update environment variables in `.env` file if needed

### Groq Integration

The system uses Groq for the language model. To modify or extend this integration:

1. Update the `initialize_llm()` method in `QAProcessor` class
2. Configure model parameters as needed
3. Update environment variables in `.env` file if needed

## Testing

### Adding New Tests

1. Create test files in a `tests` directory
2. Implement unit tests for new components and features
3. Use pytest for test execution

```python
def test_new_metric():
    evaluator = MetricsEvaluator("test_output", mock_embeddings)
    result = evaluator.calculate_new_metric("test answer", "test context", "test question")
    assert 0 <= result <= 1  # Assuming metric is normalized between 0 and 1
```

## Performance Optimization

### Embedding Optimization

1. Implement caching for embeddings to reduce API calls
2. Use batch processing for multiple documents
3. Consider local embedding models for development

### UI Optimization

1. Implement lazy loading for large visualizations
2. Use Streamlit's caching mechanisms for expensive computations
3. Optimize data structures for metrics storage and retrieval

## Deployment

### Docker Deployment

1. Create a Dockerfile in the project root
2. Define the necessary environment and dependencies
3. Configure environment variables for API keys

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
```

### Cloud Deployment

1. Configure cloud provider settings
2. Set up secure storage for API keys
3. Configure networking and security groups

## Best Practices

1. **Code Style**: Follow PEP 8 guidelines for Python code
2. **Documentation**: Document all new methods and classes
3. **Error Handling**: Implement proper error handling and logging
4. **Security**: Follow OWASP guidelines for secure coding
5. **Testing**: Write tests for all new features
6. **Version Control**: Use meaningful commit messages and branch names

## Troubleshooting

### Common Development Issues

1. **API Key Issues**: Ensure API keys are correctly set in the `.env` file
2. **Dependency Conflicts**: Use virtual environments to isolate dependencies
3. **Streamlit Caching**: Clear cache when developing new features
4. **Memory Issues**: Monitor memory usage and implement pagination for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Write tests for your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.