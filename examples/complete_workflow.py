"""
Complete Workflow Example

This example demonstrates a complete workflow using the DSPy Metric Learning package:
1. Creating a metric module
2. Collecting and labeling data
3. Optimizing the metric
4. Evaluating the optimized metric
5. Using the optimized metric for scoring new examples
"""

import dspy
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the package when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from metric_learner import (
    MetricModule,
    MetricDataManager,
    label_instances,
    optimize_metric_module,
    MetricEvaluator
)

# Sample data for demonstration
EXAMPLES = [
    {
        "input": "What is the capital of France?",
        "prediction": "Paris is the capital of France.",
        "gold": "Paris",
        "user_score": 0.9
    },
    {
        "input": "What is the tallest mountain in the world?",
        "prediction": "Mount Everest is the tallest mountain in the world.",
        "gold": "Mount Everest",
        "user_score": 0.95
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "prediction": "I believe Shakespeare wrote that play.",
        "gold": "William Shakespeare",
        "user_score": 0.7
    },
    {
        "input": "What is the boiling point of water?",
        "prediction": "Water boils at 100 degrees Celsius at sea level.",
        "gold": "100 degrees Celsius",
        "user_score": 0.85
    },
    {
        "input": "Who was the first person to walk on the moon?",
        "prediction": "Neil Armstrong was the first person to walk on the moon in 1969.",
        "gold": "Neil Armstrong",
        "user_score": 0.9
    }
]

# New examples for testing the optimized metric
TEST_EXAMPLES = [
    {
        "input": "What is the largest ocean on Earth?",
        "prediction": "The Pacific Ocean is the largest ocean on Earth.",
        "gold": "Pacific Ocean"
    },
    {
        "input": "Who painted the Mona Lisa?",
        "prediction": "Leonardo da Vinci painted the Mona Lisa in the early 16th century.",
        "gold": "Leonardo da Vinci"
    }
]


class MockLM:
    """Mock language model for demonstration purposes."""
    
    def __call__(self, prompt):
        """Return a mock score between 0.7 and 0.95."""
        import random
        return str(round(random.uniform(0.7, 0.95), 2))


def main():
    """Run the complete workflow example."""
    print("DSPy Metric Learning - Complete Workflow Example")
    print("-" * 50)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    
    # Use a mock LM for demonstration
    # In a real scenario, you would use:
    # lm = dspy.OpenAI(model="gpt-3.5-turbo")
    # dspy.settings.configure(lm=lm)
    lm = MockLM()
    
    # Create a temporary directory for this example
    example_dir = os.path.join(os.path.dirname(__file__), "example_data")
    os.makedirs(example_dir, exist_ok=True)
    
    # Create metric module and data manager
    metric_module = MetricModule(lm=lm)
    data_manager = MetricDataManager(
        metric_name="example_metric",
        data_dir=example_dir
    )
    
    # Step 2: Collect and label data
    print("\n2. Collecting and labeling data...")
    
    # In a real scenario, you might use the interactive REPL:
    # label_instances(data_manager)
    
    # For this example, we'll use predefined data
    for example in EXAMPLES:
        # Save the instance
        data_manager.save_instance(
            input=example["input"],
            prediction=example["prediction"],
            gold=example["gold"]
        )
        
        # Get the most recent instance and update its score
        instances = data_manager.load_instances()
        if instances:
            data_manager.update_user_score(
                instances[-1]["datetime"],
                example["user_score"]
            )
    
    # Step 3: Optimize the metric
    print("\n3. Optimizing the metric...")
    
    # Get the labeled dataset
    dataset = data_manager.get_labeled_dataset()
    print(f"   - Dataset size: {len(dataset)} examples")
    
    # Optimize the metric module
    optimized_module = optimize_metric_module(metric_module, dataset)
    
    # Step 4: Evaluate the optimized metric
    print("\n4. Evaluating the optimized metric...")
    
    evaluator = MetricEvaluator(optimized_module, data_manager)
    metrics = evaluator.evaluate()
    
    print(f"   - MSE: {metrics.get('mse', 'N/A')}")
    print(f"   - Correlation: {metrics.get('correlation', 'N/A')}")
    print(f"   - Number of examples: {metrics.get('num_examples', 'N/A')}")
    
    # Step 5: Use the optimized metric for new examples
    print("\n5. Using the optimized metric for new examples...")
    
    for i, example in enumerate(TEST_EXAMPLES):
        # Score with original metric
        original_score = metric_module(
            example["input"],
            example["prediction"],
            gold=example["gold"]
        )
        
        # Score with optimized metric
        optimized_score = optimized_module(
            example["input"],
            example["prediction"],
            gold=example["gold"]
        )
        
        print(f"\nExample {i+1}:")
        print(f"   - Input: {example['input']}")
        print(f"   - Prediction: {example['prediction']}")
        print(f"   - Gold: {example['gold']}")
        print(f"   - Original score: {original_score}")
        print(f"   - Optimized score: {optimized_score}")
    
    print("\nWorkflow complete!")
    
    # Cleanup
    # In a real application, you might want to keep this data
    # For this example, we'll clean up
    import shutil
    metric_dir = os.path.join(example_dir, "example_metric")
    if os.path.exists(metric_dir):
        shutil.rmtree(metric_dir)
    
    print("\nNote: This example used a mock language model.")
    print("For real applications, use an actual LLM like OpenAI's GPT models.")


if __name__ == "__main__":
    main()
