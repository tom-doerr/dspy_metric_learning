import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from metric_learner.optimization import (
    get_labeled_dataset,
    mse_metric,
    optimize_metric_module,
    MetricEvaluator
)
from metric_learner.metric_module import MetricModule
from metric_learner.data_manager import MetricDataManager

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Create a mock language model
        self.mock_lm = MagicMock()
        self.mock_lm.return_value = "0.75"
        
        # Create a mock data manager
        self.mock_data_manager = MagicMock(spec=MetricDataManager)
        
        # Create a metric module
        self.metric_module = MetricModule(lm=self.mock_lm)
    
    def test_get_labeled_dataset(self):
        """Test getting a labeled dataset from a data manager."""
        # Set up the mock data manager to return a dataset
        self.mock_data_manager.get_labeled_dataset.return_value = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Get the labeled dataset
        dataset = get_labeled_dataset(self.mock_data_manager)
        
        # Check that the data manager was called
        self.mock_data_manager.get_labeled_dataset.assert_called_once()
        
        # Check that we got the right dataset
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].input, "Q1")
        self.assertEqual(dataset[0].prediction, "A1")
        self.assertEqual(dataset[0].gold, "A1")
        self.assertEqual(dataset[0].user_score, 0.8)
    
    def test_mse_metric(self):
        """Test the MSE metric function."""
        # Create an example and prediction
        example = dspy.Example(user_score=0.8)
        pred = 0.7
        
        # Calculate the metric
        metric_value = mse_metric(example, pred)
        
        # Check that it's the negative MSE
        expected = -((0.8 - 0.7) ** 2)
        self.assertEqual(metric_value, expected)
        
        # Test with trace parameter
        metric_value_with_trace = mse_metric(example, pred, trace={})
        self.assertEqual(metric_value_with_trace, expected)
    
    @patch('dspy.teleprompt.BootstrapFewShot')
    def test_optimize_metric_module_empty_dataset(self, mock_bootstrap):
        """Test optimizing a metric module with an empty dataset."""
        # Call the function with an empty dataset
        result = optimize_metric_module(self.metric_module, [])
        
        # Check that the optimizer wasn't called
        mock_bootstrap.assert_not_called()
        
        # Check that the original module was returned
        self.assertEqual(result, self.metric_module)
    
    @patch('dspy.teleprompt.BootstrapFewShot')
    def test_optimize_metric_module(self, mock_bootstrap):
        """Test optimizing a metric module with a dataset."""
        # Set up the mock optimizer
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer
        
        # Set up a mock optimized module
        mock_optimized = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized
        
        # Create a dataset
        dataset = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Call the function
        result = optimize_metric_module(self.metric_module, dataset)
        
        # Check that the optimizer was created with the right metric
        mock_bootstrap.assert_called_once()
        args, kwargs = mock_bootstrap.call_args
        self.assertEqual(kwargs["metric"].__name__, "mse_metric")
        
        # Check that the optimizer was called with the right arguments
        mock_optimizer.compile.assert_called_once_with(self.metric_module, trainset=dataset)
        
        # Check that the optimized module was returned
        self.assertEqual(result, mock_optimized)
    
    def test_metric_evaluator_no_data(self):
        """Test the MetricEvaluator with no data."""
        # Set up the mock data manager to return an empty dataset
        self.mock_data_manager.get_labeled_dataset.return_value = []
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that we got an empty dict
        self.assertEqual(metrics, {})
    
    def test_metric_evaluator(self):
        """Test the MetricEvaluator with data."""
        # Create some examples
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.8).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.6).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that we got metrics
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("max_error", metrics)
        self.assertIn("correlation", metrics)
        self.assertIn("num_examples", metrics)
        
        # Check the number of examples
        self.assertEqual(metrics["num_examples"], 2)

if __name__ == '__main__':
    unittest.main()
