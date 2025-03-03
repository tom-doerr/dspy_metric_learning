"""
Example demonstrating the use of multiple metric functions.

This example shows how to:
1. Create multiple metric modules with different purposes
2. Use different prompt templates for different metrics
3. Manage separate data for each metric
"""

import sys
import os

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from metric_learner import (
    MetricModule,
    MetricDataManager,
    label_instances,
    optimize_metric_module
)

def main():
    # Initialize a mock language model for demonstration
    class MockLM:
        def __call__(self, prompt):
            print(f"\nPrompt sent to LM:\n{prompt}\n")
            # In a real scenario, this would call the actual LM
            if "factual accuracy" in prompt.lower():
                return "0.75"
            elif "fluency" in prompt.lower():
                return "0.9"
            elif "relevance" in prompt.lower():
                return "0.8"
            else:
                return "0.5"
    
    lm = MockLM()
    
    # Create multiple metric modules with different purposes
    accuracy_module = MetricModule(
        lm=lm,
        prompt_template=(
            "Rate the factual accuracy of the answer '{prediction}' for the question '{input}' "
            "on a scale from 0 to 1, where 0 is completely incorrect and 1 is perfectly accurate."
        )
    )
    
    fluency_module = MetricModule(
        lm=lm,
        prompt_template=(
            "Rate the fluency and readability of the answer '{prediction}' "
            "on a scale from 0 to 1, where 0 is incomprehensible and 1 is perfectly fluent."
        )
    )
    
    relevance_module = MetricModule(
        lm=lm,
        prompt_template=(
            "Rate the relevance of the answer '{prediction}' to the question '{input}' "
            "on a scale from 0 to 1, where 0 is completely irrelevant and 1 is perfectly relevant."
        )
    )
    
    # Create data managers for each metric
    accuracy_data = MetricDataManager(metric_name="accuracy")
    fluency_data = MetricDataManager(metric_name="fluency")
    relevance_data = MetricDataManager(metric_name="relevance")
    
    # Example data
    examples = [
        {
            "input": "What is the capital of France?",
            "prediction": "Paris is the capital of France.",
            "gold": "Paris"
        },
        {
            "input": "Explain quantum computing in simple terms.",
            "prediction": "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits which can only be 0 or 1. This allows quantum computers to process certain types of problems much faster than classical computers.",
            "gold": None
        }
    ]
    
    # Score and save examples with multiple metrics
    print("Scoring examples with multiple metrics...")
    for example in examples:
        print(f"\nExample: {example['input']}")
        print(f"Prediction: {example['prediction']}")
        
        # Score with accuracy metric
        accuracy_score = accuracy_module(
            example["input"],
            example["prediction"],
            gold=example.get("gold")
        )
        print(f"Accuracy score: {accuracy_score}")
        accuracy_data.save_instance(
            example["input"],
            example["prediction"],
            gold=example.get("gold"),
            score=accuracy_score
        )
        
        # Score with fluency metric
        fluency_score = fluency_module(
            example["input"],
            example["prediction"]
        )
        print(f"Fluency score: {fluency_score}")
        fluency_data.save_instance(
            example["input"],
            example["prediction"],
            score=fluency_score
        )
        
        # Score with relevance metric
        relevance_score = relevance_module(
            example["input"],
            example["prediction"]
        )
        print(f"Relevance score: {relevance_score}")
        relevance_data.save_instance(
            example["input"],
            example["prediction"],
            score=relevance_score
        )
    
    # For demo purposes, let's manually add some user scores
    accuracy_instances = accuracy_data.load_instances()
    if accuracy_instances:
        accuracy_data.update_user_score(accuracy_instances[0]["datetime"], 0.8)
        print(f"\nManually added user score 0.8 to first accuracy instance")
    
    fluency_instances = fluency_data.load_instances()
    if fluency_instances:
        fluency_data.update_user_score(fluency_instances[0]["datetime"], 0.95)
        print(f"Manually added user score 0.95 to first fluency instance")
    
    # Demonstrate how to optimize one of the metrics
    accuracy_dataset = accuracy_data.get_labeled_dataset()
    if accuracy_dataset:
        print(f"\nOptimizing accuracy metric with {len(accuracy_dataset)} labeled examples...")
        optimized_accuracy = optimize_metric_module(accuracy_module, accuracy_dataset)
        
        # Test the optimized module
        print("\nTesting optimized accuracy metric...")
        for example in examples:
            new_score = optimized_accuracy(
                example["input"],
                example["prediction"],
                gold=example.get("gold")
            )
            print(f"Example: {example['input']}")
            print(f"Optimized accuracy score: {new_score}")

if __name__ == "__main__":
    main()
