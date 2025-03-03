# DSPy Metric Learning

<div align="center">

![DSPy Metric Learning](https://img.shields.io/badge/DSPy-Metric%20Learning-blue?style=for-the-badge&logo=python&logoColor=white)

[![Test Status](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)](https://github.com/yourusername/dspy-metric-learning)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen?style=flat-square)](https://github.com/yourusername/dspy-metric-learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![DSPy](https://img.shields.io/badge/DSPy-2.0.0+-purple.svg?style=flat-square)](https://github.com/stanfordnlp/dspy)

<p align="center">
A powerful package for learning and optimizing metric functions for DSPy, leveraging language models to create better evaluation metrics for your generative AI applications.
</p>

</div>

---

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
  - [1. Creating a Metric Module](#1-creating-a-metric-module)
  - [2. Managing Data](#2-managing-data)
  - [3. Optimizing a Metric](#3-optimizing-a-metric)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 📦 Installation

<div align="center">

```bash
pip install dspy-metric-learning
```

</div>

You can also install directly from the repository:

<div align="center">

```bash
git clone https://github.com/yourusername/dspy-metric-learning.git
cd dspy-metric-learning
pip install -e .
```

</div>

## 🚀 Quick Start

<div align="center">

```python
import dspy
from metric_learner import MetricModule, MetricDataManager

# Initialize a language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")

# Create a metric module
metric = MetricModule(lm=lm)

# Score a prediction
score = metric(
    input="What is the capital of France?",
    prediction="Paris is the capital of France.",
    gold="Paris"
)

print(f"Score: {score}")  # Output: Score: 0.92
```

</div>

## 📚 Usage Examples

### 1. Creating a Metric Module

<div align="center">

```python
from metric_learner import MetricModule

# Create with custom prompt template
metric = MetricModule(
    lm=lm,
    prompt_template=(
        "Rate the factual accuracy of the answer '{prediction}' "
        "for the question '{input}' on a scale from 0 to 1."
    )
)
```

</div>

### 2. Managing Data

<div align="center">

```python
from metric_learner import MetricDataManager

# Create a data manager
data_manager = MetricDataManager(metric_name="factual_accuracy")

# Save an instance
data_manager.save_instance(
    input="What is the tallest mountain?",
    prediction="Mount Everest is the tallest mountain on Earth.",
    gold="Mount Everest",
    score=0.9
)

# Load instances
instances = data_manager.load_instances()
```

</div>

### 3. Optimizing a Metric

<div align="center">

```python
from metric_learner import optimize_metric_module

# Get labeled dataset
dataset = data_manager.get_labeled_dataset()

# Optimize the metric
optimized_metric = optimize_metric_module(metric, dataset)
```

</div>

## 🔍 Examples

<div align="center">

The `examples/` directory contains several example scripts:

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Simple demonstration of core functionality |
| `multiple_metrics.py` | Using multiple specialized metrics |
| `streamlit_app.py` | Interactive web interface for labeling and optimization |
| `complete_workflow.py` | End-to-end workflow from data collection to optimization |

</div>

Run the complete workflow example:

<div align="center">

```bash
python examples/complete_workflow.py
```

</div>

Run the Streamlit app (in headless mode):

<div align="center">

```bash
streamlit run examples/streamlit_app.py --server.headless=true
```

</div>

## 📖 API Reference

<div align="center">

### Core Components

| Component | Description |
|-----------|-------------|
| `MetricModule` | Core class for defining and using metric functions |
| `MetricDataManager` | Manages storage and retrieval of labeled instances |
| `optimize_metric_module` | Function to optimize a metric module using labeled data |
| `MetricEvaluator` | Evaluates the performance of a metric module |
| `label_instances` | Interactive REPL interface for labeling instances |

</div>

### MetricModule

```python
class MetricModule(dspy.Module):
    """Module for evaluating predictions using a language model."""
```

**Parameters:**
- `lm`: Language model to use for scoring
- `prompt_template`: Optional custom prompt template for the metric
- `demonstrations`: Optional list of demonstration examples

**Methods:**
- `__call__(input, prediction, gold=None)`: Score a prediction

### MetricDataManager

```python
class MetricDataManager:
    """Manages storage and retrieval of metric data."""
```

**Parameters:**
- `metric_name`: Name of the metric
- `data_dir`: Optional directory for storing data

**Methods:**
- `save_instance(input, prediction, gold=None, score=None)`: Save an instance
- `load_instances()`: Load all instances
- `update_user_score(datetime, score)`: Update user score for an instance
- `get_labeled_dataset()`: Get a dataset of labeled instances

### Optimization Functions

<div align="center">

```python
# Optimize a metric module
optimized_module = optimize_metric_module(
    metric_module,    # MetricModule to optimize
    dataset,          # Dataset of labeled examples
    metric_fn=None,   # Optional custom metric function
    optimizer_class=None  # Optional custom optimizer class
)

# Evaluate a metric module
evaluator = MetricEvaluator(metric_module, data_manager)
metrics = evaluator.evaluate()  # Returns MSE, correlation, etc.
```

</div>

### Interactive Labeling

<div align="center">

```python
# Start an interactive labeling session
label_instances(
    data_manager,     # MetricDataManager instance
    quit_after=None,  # Optional number of instances to label
    skip_labeled=True # Whether to skip already labeled instances
)
```

</div>

## 🧪 Testing

<div align="center">

Run unit tests:

```bash
python -m pytest tests/
```

Run integration tests:

```bash
python -m pytest integration_tests/
```

Run specific test categories:

```bash
python -m pytest -m "integration and not slow"
```

</div>

## 👥 Contributing

<div align="center">

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

</div>

## 📄 License

<div align="center">

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<img src="https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20using-DSPy-purple?style=for-the-badge&logo=python&logoColor=white" alt="Made with ❤️ using DSPy">

</div>
