# DSPy Metric Learning

[![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/yourusername/dspy-metric-learning)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/yourusername/dspy-metric-learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful package for learning and optimizing metric functions for DSPy, leveraging language models to create better evaluation metrics for your generative AI applications.

## 🌟 Features

- **LLM-based Evaluation**: Define metric functions as DSPy modules using language models
- **Custom Scoring**: Pass your preferred language models for rating predictions
- **Data Management**: Store and manage scored outputs in an organized directory structure
- **Interactive Labeling**: Simple REPL interface for human labeling of examples
- **Optimization**: DSPy-powered optimization for metric function modules
- **Multi-metric Support**: Create and manage multiple specialized metric functions
- **Comprehensive Testing**: Extensive test suite with 92% code coverage

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [Multiple Metrics](#multiple-metrics)
  - [Optimization](#optimization)
  - [Interactive Labeling](#interactive-labeling)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Using pip

```bash
pip install dspy-metric-learning
```

### Using Poetry

```bash
poetry add dspy-metric-learning
```

## 🚀 Quick Start

```python
import dspy
from metric_learner import MetricModule, MetricDataManager

# Initialize with your preferred language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Create a metric module
metric = MetricModule(lm=lm)

# Score a prediction
score = metric("What is the capital of France?", "Paris")
print(f"Score: {score}")  # Output: Score: 0.95 (or similar)
```

## 📚 Usage Examples

### Basic Usage

```python
import dspy
from metric_learner import MetricModule, MetricDataManager

# Initialize with your preferred language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Create a metric module and data manager
metric_module = MetricModule(lm=lm)
data_manager = MetricDataManager(metric_name="my_metric")

# Score a prediction
input_text = "What is the capital of France?"
prediction = "Paris"
score = metric_module(input_text, prediction, gold="Paris")
print(f"Score: {score}")

# Save the instance for later use
data_manager.save_instance(input_text, prediction, gold="Paris", score=score)
```

### Multiple Metrics

You can create and manage multiple specialized metric functions:

```python
# Create separate metric modules and data managers
accuracy_metric = MetricModule(lm=lm, metric_name="accuracy")
fluency_metric = MetricModule(lm=lm, metric_name="fluency")
relevance_metric = MetricModule(lm=lm, metric_name="relevance")

# Use them independently
accuracy_score = accuracy_metric(input_text, prediction, gold=gold)
fluency_score = fluency_metric(input_text, prediction)
relevance_score = relevance_metric(input_text, prediction)

# Combine scores as needed
combined_score = (accuracy_score + fluency_score + relevance_score) / 3
```

### Optimization

Optimize your metric functions using human-labeled examples:

```python
from metric_learner import optimize_metric_module

# Get labeled dataset
dataset = data_manager.get_labeled_dataset()

# Optimize the metric module
optimized_module = optimize_metric_module(metric_module, dataset)

# Use the optimized module
new_score = optimized_module(input_text, prediction, gold="Paris")
print(f"Optimized score: {new_score}")

# Evaluate the optimized metric
from metric_learner import MetricEvaluator
evaluator = MetricEvaluator(optimized_module, data_manager)
metrics = evaluator.evaluate()
print(f"MSE: {metrics['mse']}, Correlation: {metrics['correlation']}")
```

### Interactive Labeling

Label examples interactively using the REPL interface:

```python
from metric_learner import label_instances

# Start the interactive labeling session
label_instances(data_manager)
```

This will present each unlabeled instance and prompt for a human score.

## 🔍 Examples

The `examples/` directory contains several example scripts:

- `basic_usage.py`: Simple demonstration of core functionality
- `multiple_metrics.py`: Using multiple specialized metrics
- `streamlit_app.py`: Interactive web interface for labeling and optimization
- `complete_workflow.py`: End-to-end workflow from data collection to optimization

Run the complete workflow example:

```bash
python examples/complete_workflow.py
```

Run the Streamlit app (in headless mode):

```bash
streamlit run examples/streamlit_app.py --server.headless=true
```

## 📖 API Reference

### MetricModule

The core class for creating and using metric functions.

```python
metric = MetricModule(
    lm,                      # Language model to use
    metric_name="default",   # Name of the metric
    prompt_template=None,    # Optional custom prompt template
    demonstrations=[]        # Optional few-shot examples
)
```

### MetricDataManager

Manages storage and retrieval of metric instances.

```python
manager = MetricDataManager(
    metric_name,            # Name of the metric
    data_dir=".metrics_data" # Base directory for storing data
)
```

### Optimization Functions

```python
# Optimize a metric module using labeled data
optimized_module = optimize_metric_module(
    metric_module,          # Module to optimize
    dataset,                # Labeled dataset
    metric_fn=None,         # Optional custom metric function
    optimizer_class=None    # Optional custom optimizer class
)

# Evaluate a metric module
evaluator = MetricEvaluator(metric_module, data_manager)
metrics = evaluator.evaluate()
```

## 🧪 Testing

The package includes comprehensive test suites:

### Unit Tests

Run unit tests with:

```bash
python -m pytest tests/
```

### Integration Tests

Integration tests are in a separate directory and can be run with:

```bash
python -m pytest integration_tests/
```

You can also run tests based on markers:

```bash
# Run only slow tests
python -m pytest -m "slow"

# Run all tests except slow ones
python -m pytest -m "not slow"

# Run only integration tests
python -m pytest -m "integration"
```

### Code Coverage

The package has excellent test coverage (92%). Run tests with coverage reporting:

```bash
python -m pytest --cov=metric_learner
```

For a detailed HTML report:

```bash
python -m pytest --cov=metric_learner --cov-report=html
```

See the [integration tests README](integration_tests/README.md) for more details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
