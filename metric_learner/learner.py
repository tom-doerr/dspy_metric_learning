import dspy
from typing import List, Callable, Optional, Union, Any

from .metric_module import MetricModule
from .optimization import optimize_metric_module


class MetricLearner:
    """
    A class that handles the optimization of a DSPy program using a metric module.
    
    This class provides a simplified interface for optimizing DSPy programs
    with custom metric functions.
    """
    
    def __init__(
        self,
        program: dspy.Module,
        metric_module: MetricModule,
        num_iterations: int = 5,
        optimizer_class: Optional[Any] = None,
        verbose: bool = False
    ):
        """
        Initialize the MetricLearner.
        
        Args:
            program: The DSPy program to optimize
            metric_module: The MetricModule instance to use for optimization
            num_iterations: Number of optimization iterations to run
            optimizer_class: Optional custom optimizer class (defaults to BootstrapFewShot)
            verbose: Whether to print detailed progress information
        """
        self.program = program
        self.metric_module = metric_module
        self.num_iterations = num_iterations
        self.optimizer_class = optimizer_class or dspy.teleprompt.BootstrapFewShot
        self.verbose = verbose
        
    def _create_metric_fn(self) -> Callable:
        """
        Create a metric function for the DSPy optimizer.
        
        Returns:
            Callable: A metric function compatible with DSPy optimizers
        """
        def metric_fn(example, pred, trace=None):
            """
            Metric function for DSPy optimization.
            
            Args:
                example: Example with gold answer
                pred: Prediction from the model
                trace: Optional trace object
                
            Returns:
                float: Metric score (higher is better)
            """
            # Extract the answer from the prediction
            predicted_answer = pred.answer if hasattr(pred, 'answer') else str(pred)
            
            # Extract the gold answer from the example
            gold_answer = example.answer if hasattr(example, 'answer') else ""
            
            # Extract the question from the example
            question = example.question if hasattr(example, 'question') else ""
            
            # Use the metric module to score the prediction
            score = self.metric_module(question, predicted_answer, gold=gold_answer)
            
            if self.verbose:
                print(f"Question: {question}")
                print(f"Predicted: {predicted_answer}")
                print(f"Gold: {gold_answer}")
                print(f"Score: {score}")
                print("-" * 50)
            
            return score
        
        return metric_fn
    
    def optimize(self, examples: List[dspy.Example]) -> dspy.Module:
        """
        Optimize the DSPy program using the provided examples.
        
        Args:
            examples: List of DSPy examples to use for optimization
            
        Returns:
            dspy.Module: The optimized DSPy program
        """
        if self.verbose:
            print(f"Starting optimization with {len(examples)} examples...")
            print(f"Number of iterations: {self.num_iterations}")
        
        # Create the optimizer
        optimizer = self.optimizer_class(
            metric=self._create_metric_fn(),
            max_bootstrapped_demos=self.num_iterations
        )
        
        # Optimize the program
        optimized_program = optimizer.compile(self.program, trainset=examples)
        
        if self.verbose:
            print("Optimization complete.")
        
        return optimized_program
