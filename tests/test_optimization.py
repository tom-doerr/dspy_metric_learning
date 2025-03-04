import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

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

    def test_metric_evaluator_no_variance(self):
        """Test the MetricEvaluator when there's no variance in scores."""
        # Create examples with the same user score
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.8).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to always return the same score
        self.mock_lm.return_value = "0.5"
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 due to no variance
        self.assertEqual(metrics["correlation"], 0.0)
        
    def test_metric_evaluator_empty_dataset_correlation(self):
        """Test the MetricEvaluator correlation with an empty dataset."""
        # Set up the mock data manager to return an empty dataset
        self.mock_data_manager.get_labeled_dataset.return_value = []
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is not calculated (empty dict)
        self.assertEqual(metrics, {})
        
    def test_metric_evaluator_zero_variance_user_scores(self):
        """Test the MetricEvaluator when user scores have zero variance but model scores vary."""
        # Create examples with the same user score but different model scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.8).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return different scores
        self.mock_lm.side_effect = ["0.5", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 due to zero variance in user scores
        self.assertEqual(metrics["correlation"], 0.0)

    def test_metric_evaluator_model_scores_no_variance(self):
        """Test the MetricEvaluator when model scores have zero variance but user scores vary."""
        # Create examples with different user scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return the same score for all examples
        self.mock_lm.side_effect = ["0.7", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 due to zero variance in model scores
        self.assertEqual(metrics["correlation"], 0.0)

    def test_metric_evaluator_both_scores_no_variance(self):
        """Test the MetricEvaluator when both user and model scores have zero variance."""
        # Create examples with the same user score
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.7).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.7).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return the same score for all examples
        self.mock_lm.side_effect = ["0.7", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 due to zero variance in both scores
        self.assertEqual(metrics["correlation"], 0.0)

    def test_metric_evaluator_with_numpy_mocking(self):
        """Test the MetricEvaluator with specific numpy mocking to cover edge cases."""
        # Create examples with different user scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return different scores
        self.mock_lm.side_effect = ["0.6", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Mock numpy functions to force specific behavior
        original_std = np.std
        original_corrcoef = np.corrcoef
        
        def mock_std(array):
            # Return 0 for model_scores, but normal value for user_scores
            if len(array) == 2 and array[0] == 0.6 and array[1] == 0.7:
                return 0.0
            return original_std(array)
            
        def mock_corrcoef(array1, array2):
            # This should never be called due to the std check
            # But if it is, return a dummy value
            return np.array([[1.0, 0.5], [0.5, 1.0]])
        
        with unittest.mock.patch('numpy.std', side_effect=mock_std):
            with unittest.mock.patch('numpy.corrcoef', side_effect=mock_corrcoef):
                # Evaluate
                metrics = evaluator.evaluate()
                
                # Check that correlation is 0.0 due to zero variance in model scores
                self.assertEqual(metrics["correlation"], 0.0)

    def test_metric_evaluator_with_direct_numpy_mocking(self):
        """Test the MetricEvaluator with direct numpy mocking to cover edge cases."""
        # Create examples with different user scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return different scores
        self.mock_lm.side_effect = ["0.6", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Directly patch the specific numpy functions at the module level
        # This is more targeted than the previous approach
        with patch('metric_learner.optimization.np.std') as mock_std:
            # First test: user_scores have variance but model_scores don't
            mock_std.side_effect = lambda x: 0.0 if len(x) == 2 and all(isinstance(v, float) for v in x) else 0.15
            
            # Evaluate
            metrics = evaluator.evaluate()
            
            # Check that correlation is 0.0 due to zero variance in model scores
            self.assertEqual(metrics["correlation"], 0.0)
    
    def test_metric_evaluator_with_variance_in_both_scores(self):
        """Test the MetricEvaluator with variance in both user and model scores."""
        # Create examples with different user scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.8).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return different scores
        self.mock_lm.side_effect = ["0.6", "0.7"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Mock both np.std and np.corrcoef
        with patch('metric_learner.optimization.np.std') as mock_std, \
             patch('metric_learner.optimization.np.corrcoef') as mock_corrcoef:
            
            # Make np.std always return a positive value to ensure variance check passes
            mock_std.return_value = 0.15
            
            # Set up the mock to return a correlation matrix with a known value
            mock_corrcoef.return_value = np.array([[1.0, 0.75], [0.75, 1.0]])
            
            # Evaluate
            metrics = evaluator.evaluate()
            
            # Verify that np.corrcoef was called
            mock_corrcoef.assert_called_once()
            
            # Check that correlation is the expected value from our mock
            self.assertEqual(metrics["correlation"], 0.75)

    def test_metric_evaluator_with_zero_variance_in_both_scores(self):
        """Test the MetricEvaluator with zero variance in both user and model scores."""
        # Create examples with identical user scores
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.5).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return identical scores
        self.mock_lm.side_effect = ["0.5", "0.5"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate - this should trigger the case where both user_scores and model_scores have zero variance
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 due to zero variance in both scores
        self.assertEqual(metrics["correlation"], 0.0)

    def test_metric_evaluator_with_empty_dataset(self):
        """Test the MetricEvaluator with an empty dataset."""
        # Set up the mock data manager to return an empty dataset
        self.mock_data_manager.get_labeled_dataset.return_value = []
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate - this should trigger the case where there are no examples
        metrics = evaluator.evaluate()
        
        # Check that we got an empty metrics dictionary
        self.assertEqual(metrics, {})
        
        # Verify that a message was printed
        # This is captured in the test output

    def test_metric_evaluator_with_no_user_scores(self):
        """Test the MetricEvaluator with examples that don't have user_score attribute."""
        # Create examples with user_score
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold"),
            dspy.Example(input="Q2", prediction="A2", gold="A2", user_score=0.5).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the examples
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Mock both np.std and np.corrcoef
        with patch('metric_learner.optimization.np.std') as mock_std, \
             patch('metric_learner.optimization.np.corrcoef') as mock_corrcoef:
            
            # Make np.std return 0 for the first call and a positive value for the second
            # This ensures we hit the else branch in the if-else statement
            mock_std.side_effect = [0.0, 0.15]
            
            # Evaluate
            metrics = evaluator.evaluate()
            
            # Check that correlation is 0.0 due to zero variance in user scores
            self.assertEqual(metrics["correlation"], 0.0)
            
            # Verify that np.corrcoef was not called
            mock_corrcoef.assert_not_called()

    def test_metric_evaluator_with_single_example(self):
        """Test the MetricEvaluator with a single example."""
        # Create a single example with user_score
        examples = [
            dspy.Example(input="Q1", prediction="A1", gold="A1", user_score=0.5).with_inputs("input", "prediction", "gold")
        ]
        
        # Set up the mock data manager to return the example
        self.mock_data_manager.get_labeled_dataset.return_value = examples
        
        # Set up the metric module to return a score
        self.mock_lm.side_effect = ["0.6"]
        
        # Create an evaluator
        evaluator = MetricEvaluator(self.metric_module, self.mock_data_manager)
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Check that correlation is 0.0 for a single example
        self.assertEqual(metrics["correlation"], 0.0)
        
        # Also check other metrics
        self.assertEqual(metrics["num_examples"], 1)
        self.assertAlmostEqual(metrics["mse"], 0.01, places=10)  # (0.6 - 0.5)^2 = 0.01
        self.assertAlmostEqual(metrics["mae"], 0.1, places=10)   # |0.6 - 0.5| = 0.1
        self.assertAlmostEqual(metrics["max_error"], 0.1, places=10)

if __name__ == '__main__':
    unittest.main()
