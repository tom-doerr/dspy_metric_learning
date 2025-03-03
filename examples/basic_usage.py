"""
Basic usage example for the DSPy Metric Learning package.

This example demonstrates how to:
1. Create a metric module
2. Score predictions
3. Label instances
4. Optimize the metric module
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
    optimize_metric_module,
    get_labeled_dataset
)

def main():
    # Initialize a language model (use your preferred model)
    # This example uses a mock LM for demonstration
    class MockLM:
        def __call__(self, prompt):
            print(f"\nPrompt sent to LM:\n{prompt}\n")
            # In a real scenario, this would call the actual LM
            return "0.85"
    
    lm = MockLM()
    
    # Create a metric module and data manager
    metric_module = MetricModule(lm=lm)
    data_manager = MetricDataManager(metric_name="example_metric")
    
    # Example data
    examples = [
        {
            "input": "What is the capital of France?",
            "prediction": "Paris",
            "gold": "Paris"
        },
        {
            "input": "What is the largest planet in our solar system?",
            "prediction": "Jupiter is the largest planet in our solar system.",
            "gold": "Jupiter"
        },
        {
            "input": "Who wrote 'Pride and Prejudice'?",
            "prediction": "I think it was Jane Eyre.",
            "gold": "Jane Austen"
        }
    ]
    
    # Score and save examples
    print("Scoring examples...")
    for example in examples:
        score = metric_module(
            example["input"],
            example["prediction"],
            gold=example["gold"]
        )
        print(f"Example: {example['input']}")
        print(f"Prediction: {example['prediction']}")
        print(f"Score: {score}")
        
        data_manager.save_instance(
            example["input"],
            example["prediction"],
            gold=example["gold"],
            score=score
        )
    
    # Label instances (commented out for automated demo)
    # print("\nStarting labeling session...")
    # label_instances(data_manager)
    
    # For demo purposes, let's manually add some user scores
    instances = data_manager.load_instances()
    if instances:
        # Add a user score to the first instance
        data_manager.update_user_score(instances[0]["datetime"], 0.9)
        print(f"\nManually added user score 0.9 to first instance")
        
        # Add a user score to the second instance
        if len(instances) > 1:
            data_manager.update_user_score(instances[1]["datetime"], 0.7)
            print(f"Manually added user score 0.7 to second instance")
    
    # Get labeled dataset
    dataset = get_labeled_dataset(data_manager)
    print(f"\nFound {len(dataset)} labeled examples")
    
    # Optimize the metric module
    if dataset:
        print("\nOptimizing metric module...")
        optimized_module = optimize_metric_module(metric_module, dataset)
        
        # Test the optimized module
        print("\nTesting optimized module...")
        for example in examples:
            new_score = optimized_module(
                example["input"],
                example["prediction"],
                gold=example["gold"]
            )
            print(f"Example: {example['input']}")
            print(f"Optimized score: {new_score}")
    else:
        print("\nNo labeled data available for optimization")

if __name__ == "__main__":
    main()
