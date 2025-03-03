import dspy
import numpy as np

def get_labeled_dataset(data_manager):
    """
    Get a labeled dataset from a data manager.
    
    Args:
        data_manager: MetricDataManager instance
        
    Returns:
        list: List of dspy.Example objects
    """
    return data_manager.get_labeled_dataset()

def mse_metric(example, pred, trace=None):
    """
    Mean squared error metric for optimization.
    
    Args:
        example: Example with user_score
        pred: Predicted score
        trace: Optional trace object (ignored)
        
    Returns:
        float: Negative MSE (higher is better for DSPy optimizers)
    """
    predicted_score = pred
    user_score = example.user_score
    return -(predicted_score - user_score) ** 2

def optimize_metric_module(metric_module, dataset, metric_fn=None, optimizer_class=None):
    """
    Optimize a metric module using labeled data.
    
    Args:
        metric_module: MetricModule instance to optimize
        dataset: List of dspy.Example objects with user scores
        metric_fn: Optional custom metric function
        optimizer_class: Optional custom optimizer class
        
    Returns:
        MetricModule: Optimized metric module
    """
    if not dataset:
        print("No labeled data available for optimization.")
        return metric_module
    
    # Use default metric function if none provided
    if metric_fn is None:
        metric_fn = mse_metric
    
    # Use default optimizer if none provided
    if optimizer_class is None:
        optimizer_class = dspy.teleprompt.BootstrapFewShot
    
    # Create and configure the optimizer
    optimizer = optimizer_class(metric=metric_fn)
    
    # Compile the metric module
    print(f"Optimizing metric module with {len(dataset)} labeled examples...")
    optimized_module = optimizer.compile(metric_module, trainset=dataset)
    
    # Return the optimized module
    print("Optimization complete.")
    return optimized_module


class MetricEvaluator:
    """
    Evaluates the performance of a metric module against user scores.
    """
    
    def __init__(self, metric_module, data_manager):
        """
        Initialize the evaluator.
        
        Args:
            metric_module: MetricModule instance
            data_manager: MetricDataManager instance
        """
        self.metric_module = metric_module
        self.data_manager = data_manager
    
    def evaluate(self):
        """
        Evaluate the metric module against labeled data.
        
        Returns:
            dict: Evaluation metrics
        """
        # Get labeled data
        dataset = self.data_manager.get_labeled_dataset()
        
        if not dataset:
            print("No labeled data available for evaluation.")
            return {}
        
        # Calculate metrics
        errors = []
        for example in dataset:
            # Get model prediction
            pred_score = self.metric_module(
                example.input, 
                example.prediction, 
                gold=example.gold
            )
            
            # Calculate error
            error = pred_score - example.user_score
            errors.append(error)
        
        # Convert to numpy array
        errors = np.array(errors)
        
        # Calculate metrics
        metrics = {
            "mse": float(np.mean(errors ** 2)),
            "mae": float(np.mean(np.abs(errors))),
            "max_error": float(np.max(np.abs(errors))),
            "num_examples": len(dataset)
        }
        
        # Only calculate correlation if we have more than one example
        if len(dataset) > 1:
            user_scores = np.array([e.user_score for e in dataset])
            model_scores = np.array([self.metric_module(e.input, e.prediction, gold=e.gold) for e in dataset])
            
            # Check if there's variance in both arrays to avoid division by zero
            if np.std(user_scores) > 0 and np.std(model_scores) > 0:
                metrics["correlation"] = float(np.corrcoef(user_scores, model_scores)[0, 1])
            else:
                metrics["correlation"] = 0.0
        else:
            metrics["correlation"] = 0.0
        
        return metrics
