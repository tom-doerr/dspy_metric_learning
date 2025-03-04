#!/usr/bin/env python3
"""
Example of using DSPy Metric Learning with OpenRouter's Gemini model.
"""

import dspy
from metric_learner import MetricLearner, MetricModule

# Configure DSPy to use OpenRouter with Gemini model
# API key is loaded automatically from environment
lm = dspy.LM(model="openrouter/google/gemini-2.0-flash-001")
dspy.settings.configure(lm=lm)

# Create a MetricModule with custom prompt template
metric_module = MetricModule(
    lm=lm,
    prompt_template=(
        "Rate the quality of the answer '{prediction}' for the question '{input}' "
        "on a scale from 0 to 1, where 0 is completely incorrect and 1 is perfect. "
        "If the gold answer '{gold}' is provided, check if the prediction contains this information."
    )
)

# Sample data for optimization
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
]

# Define a simple DSPy program to optimize
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)

# Create and train the metric learner
metric_learner = MetricLearner(
    program=SimpleQA(),
    metric_module=metric_module,
    num_iterations=2,  # Small number for demonstration
    verbose=True
)

# Prepare the data
train_examples = [
    dspy.Example(
        question=item["question"],
        answer=item["answer"]
    ).with_inputs("question") for item in train_data
]

# Run the optimization
optimized_program = metric_learner.optimize(train_examples)

# Test the optimized program
test_question = "What is the capital of Italy?"
result = optimized_program(question=test_question)
print(f"\nQuestion: {test_question}")
print(f"Answer: {result.answer}")

# You can also inspect the learned metric function
print("\nLearned metric function:")
print(metric_module.get_learned_metric_fn())
