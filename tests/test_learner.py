import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from metric_learner.learner import MetricLearner
from metric_learner.metric_module import MetricModule

class TestMetricLearner(unittest.TestCase):
    def setUp(self):
        # Create a mock language model
        self.mock_lm = MagicMock()
        self.mock_lm.return_value = "0.75"
        
        # Create a metric module with the mock LM
        self.metric_module = MetricModule(lm=self.mock_lm)
        
        # Create a mock DSPy program
        self.mock_program = MagicMock(spec=dspy.Module)
        # Ensure the program is not compiled
        self.mock_program._compiled = False
        
        # Create a MetricLearner
        self.learner = MetricLearner(
            program=self.mock_program,
            metric_module=self.metric_module,
            num_iterations=3,
            verbose=True
        )
    
    def test_initialization(self):
        """Test that the MetricLearner initializes correctly."""
        self.assertEqual(self.learner.program, self.mock_program)
        self.assertEqual(self.learner.metric_module, self.metric_module)
        self.assertEqual(self.learner.num_iterations, 3)
        self.assertEqual(self.learner.optimizer_class, dspy.teleprompt.BootstrapFewShot)
        self.assertTrue(self.learner.verbose)
        
        # Test with custom optimizer class
        mock_optimizer_class = MagicMock()
        custom_learner = MetricLearner(
            program=self.mock_program,
            metric_module=self.metric_module,
            optimizer_class=mock_optimizer_class
        )
        self.assertEqual(custom_learner.optimizer_class, mock_optimizer_class)
    
    def test_create_metric_fn(self):
        """Test the _create_metric_fn method."""
        # Get the metric function
        metric_fn = self.learner._create_metric_fn()
        
        # Create a mock example and prediction
        example = dspy.Example(question="What is 2+2?", answer="4")
        pred = MagicMock()
        pred.answer = "4"
        
        # Call the metric function
        score = metric_fn(example, pred)
        
        # Check that the metric module was called with the right arguments
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
    
    def test_optimize(self):
        """Test the optimize method."""
        # Create some examples
        examples = [
            dspy.Example(question="What is 2+2?", answer="4"),
            dspy.Example(question="What is 3+3?", answer="6")
        ]
        
        # Mock the optimizer class and instance
        mock_optimizer = MagicMock()
        mock_optimized_program = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized_program
        
        # Replace the optimizer_class with our mock
        self.learner.optimizer_class = MagicMock(return_value=mock_optimizer)
        
        # Call the optimize method
        result = self.learner.optimize(examples)
        
        # Check that the optimizer was created with the right arguments
        self.learner.optimizer_class.assert_called_once()
        args, kwargs = self.learner.optimizer_class.call_args
        self.assertEqual(kwargs["max_bootstrapped_demos"], 3)
        
        # Check that the optimizer was called with the right arguments
        mock_optimizer.compile.assert_called_once_with(self.mock_program, trainset=examples)
        
        # Check that the optimized program was returned
        self.assertEqual(result, mock_optimized_program)
        
    def test_create_metric_fn_with_string_pred(self):
        """Test the _create_metric_fn method with a string prediction."""
        # Get the metric function
        metric_fn = self.learner._create_metric_fn()
        
        # Create a mock example
        example = dspy.Example(question="What is 2+2?", answer="4")
        
        # Call the metric function with a string prediction
        score = metric_fn(example, "4")
        
        # Check that the metric module was called with the right arguments
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
        
    def test_create_metric_fn_with_missing_attributes(self):
        """Test the _create_metric_fn method with examples missing attributes."""
        # Get the metric function
        metric_fn = self.learner._create_metric_fn()
        
        # Create an example without question or answer
        example = dspy.Example(input="2+2")
        pred = MagicMock()
        pred.output = "4"
        
        # Call the metric function
        score = metric_fn(example, pred)
        
        # Check that the score is still returned
        self.assertEqual(score, 0.75)
        
    def test_optimize_with_empty_examples(self):
        """Test the optimize method with empty examples."""
        # Create an empty list of examples
        examples = []
        
        # Mock the optimizer class and instance
        mock_optimizer = MagicMock()
        mock_optimized_program = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized_program
        
        # Replace the optimizer_class with our mock
        self.learner.optimizer_class = MagicMock(return_value=mock_optimizer)
        
        # Call the optimize method
        result = self.learner.optimize(examples)
        
        # Check that the optimizer was created with the right arguments
        self.learner.optimizer_class.assert_called_once()
        
        # Check that the optimizer was called with the right arguments
        mock_optimizer.compile.assert_called_once_with(self.mock_program, trainset=examples)
        
        # Check that the optimized program was returned
        self.assertEqual(result, mock_optimized_program)
        
    def test_optimize_with_custom_num_iterations(self):
        """Test the optimize method with a custom number of iterations."""
        # Create some examples
        examples = [
            dspy.Example(question="What is 2+2?", answer="4"),
            dspy.Example(question="What is 3+3?", answer="6")
        ]
        
        # Create a learner with a custom number of iterations
        learner = MetricLearner(
            program=self.mock_program,
            metric_module=self.metric_module,
            num_iterations=10,
            verbose=False
        )
        
        # Mock the optimizer class and instance
        mock_optimizer = MagicMock()
        mock_optimized_program = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized_program
        
        # Replace the optimizer_class with our mock
        learner.optimizer_class = MagicMock(return_value=mock_optimizer)
        
        # Call the optimize method
        result = learner.optimize(examples)
        
        # Check that the optimizer was created with the right arguments
        learner.optimizer_class.assert_called_once()
        args, kwargs = learner.optimizer_class.call_args
        self.assertEqual(kwargs["max_bootstrapped_demos"], 10)
        
    def test_verbose_output(self):
        """Test that verbose output works correctly."""
        # Create a learner with verbose=True
        learner = MetricLearner(
            program=self.mock_program,
            metric_module=self.metric_module,
            num_iterations=3,
            verbose=True
        )
        
        # Get the metric function
        metric_fn = learner._create_metric_fn()
        
        # Create a mock example and prediction
        example = dspy.Example(question="What is 2+2?", answer="4")
        pred = MagicMock()
        pred.answer = "4"
        
        # Patch the print function to capture output
        with patch('builtins.print') as mock_print:
            # Call the metric function
            score = metric_fn(example, pred)
            
            # Check that print was called with the expected arguments
            mock_print.assert_any_call("Question: What is 2+2?")
            mock_print.assert_any_call("Predicted: 4")
            mock_print.assert_any_call("Gold: 4")
            mock_print.assert_any_call("Score: 0.75")
            
    def test_non_verbose_output(self):
        """Test that non-verbose output works correctly."""
        # Create a learner with verbose=False
        learner = MetricLearner(
            program=self.mock_program,
            metric_module=self.metric_module,
            num_iterations=3,
            verbose=False
        )
        
        # Get the metric function
        metric_fn = learner._create_metric_fn()
        
        # Create a mock example and prediction
        example = dspy.Example(question="What is 2+2?", answer="4")
        pred = MagicMock()
        pred.answer = "4"
        
        # Patch the print function to capture output
        with patch('builtins.print') as mock_print:
            # Call the metric function
            score = metric_fn(example, pred)
            
            # Check that print was not called
            mock_print.assert_not_called()

if __name__ == '__main__':
    unittest.main()
