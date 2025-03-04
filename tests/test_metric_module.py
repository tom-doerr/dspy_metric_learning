import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_learner.metric_module import MetricModule

class TestMetricModule(unittest.TestCase):
    def setUp(self):
        # Create a mock language model
        self.mock_lm = MagicMock()
        self.mock_lm.return_value = "0.75"
        
        # Create a metric module with the mock LM
        self.metric_module = MetricModule(lm=self.mock_lm)
    
    def test_initialization(self):
        """Test that the MetricModule initializes correctly."""
        self.assertEqual(self.metric_module.lm, self.mock_lm)
        self.assertIsNotNone(self.metric_module.prompt_template)
        self.assertEqual(len(self.metric_module.demonstrations), 0)
    
    def test_custom_prompt_template(self):
        """Test that a custom prompt template is used correctly."""
        custom_template = "Rate the answer '{prediction}' for '{input}' from 0 to 1."
        metric_module = MetricModule(lm=self.mock_lm, prompt_template=custom_template)
        self.assertEqual(metric_module.prompt_template, custom_template)
    
    def test_forward_basic(self):
        """Test the forward method with basic input."""
        # Call the forward method
        score = self.metric_module("What is 2+2?", "4")
        
        # Check that the LM was called with the right prompt
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
    
    def test_forward_with_gold(self):
        """Test the forward method with a gold answer."""
        # Call the forward method
        score = self.metric_module("What is 2+2?", "4", gold="4")
        
        # Check that the LM was called with the right prompt
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
    
    def test_forward_with_demonstrations(self):
        """Test the forward method with demonstrations."""
        # Add some demonstrations
        self.metric_module.demonstrations = [
            {"input": "What is 1+1?", "prediction": "2", "gold": "2", "score": 1.0},
            {"input": "What is 3+3?", "prediction": "7", "gold": "6", "score": 0.0}
        ]
        
        # Call the forward method
        score = self.metric_module("What is 2+2?", "4")
        
        # Check that the LM was called with the right prompt
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
    
    def test_parse_score_valid(self):
        """Test the _parse_score method with a valid score."""
        # Test with a valid score
        score = self.metric_module._parse_score("0.75")
        self.assertEqual(score, 0.75)
        
        # Test with a valid score in a sentence
        score = self.metric_module._parse_score("The score is 0.75 out of 1.")
        self.assertEqual(score, 0.75)
    
    def test_parse_score_invalid(self):
        """Test the _parse_score method with an invalid score."""
        # Test with an invalid score
        score = self.metric_module._parse_score("Not a number")
        self.assertEqual(score, 0.5)  # Default score
    
    def test_parse_score_out_of_range_high(self):
        """Test the _parse_score method with a score > 1."""
        # Test with a score > 1
        score = self.metric_module._parse_score("1.5")
        self.assertEqual(score, 1.0)  # Capped at 1.0
    
    def test_parse_score_out_of_range_low(self):
        """Test the _parse_score method with a score < 0."""
        # Test with a score < 0
        score = self.metric_module._parse_score("-0.5")
        self.assertEqual(score, 0.0)  # Capped at 0.0
    
    def test_get_learned_metric_fn(self):
        """Test the get_learned_metric_fn method."""
        # Add some demonstrations
        self.metric_module.demonstrations = [
            {"input": "What is 1+1?", "prediction": "2", "gold": "2", "score": 1.0},
            {"input": "What is 3+3?", "prediction": "7", "gold": "6", "score": 0.0}
        ]
        
        # Get the learned metric function
        learned_fn = self.metric_module.get_learned_metric_fn()
        
        # Check that it contains the expected information
        self.assertIn("Prompt Template:", learned_fn)
        self.assertIn("Demonstrations:", learned_fn)
        self.assertIn("What is 1+1?", learned_fn)
        self.assertIn("What is 3+3?", learned_fn)
    
    def test_error_handling_lm_failure(self):
        """Test error handling when the LM fails."""
        # Create a mock LM that raises an exception
        error_lm = MagicMock(side_effect=Exception("LM failure"))
        
        # Create a metric module with the error LM
        error_metric_module = MetricModule(lm=error_lm)
        
        # Call the forward method
        score = error_metric_module("What is 2+2?", "4")
        
        # Check that a default score is returned
        self.assertEqual(score, 0.5)  # Default score
    
    def test_add_demonstration(self):
        """Test adding a demonstration to the metric module."""
        # Initial state
        self.assertEqual(len(self.metric_module.demonstrations), 0)
        
        # Add a demonstration
        self.metric_module.add_demonstration(
            input="What is 1+1?",
            prediction="2",
            gold="2",
            score=1.0
        )
        
        # Check that the demonstration was added
        self.assertEqual(len(self.metric_module.demonstrations), 1)
        self.assertEqual(self.metric_module.demonstrations[0]["input"], "What is 1+1?")
        self.assertEqual(self.metric_module.demonstrations[0]["prediction"], "2")
        self.assertEqual(self.metric_module.demonstrations[0]["gold"], "2")
        self.assertEqual(self.metric_module.demonstrations[0]["score"], 1.0)
    
    def test_clear_demonstrations(self):
        """Test clearing demonstrations from the metric module."""
        # Add some demonstrations
        self.metric_module.demonstrations = [
            {"input": "What is 1+1?", "prediction": "2", "gold": "2", "score": 1.0},
            {"input": "What is 3+3?", "prediction": "7", "gold": "6", "score": 0.0}
        ]
        
        # Initial state
        self.assertEqual(len(self.metric_module.demonstrations), 2)
        
        # Clear demonstrations
        self.metric_module.clear_demonstrations()
        
        # Check that demonstrations were cleared
        self.assertEqual(len(self.metric_module.demonstrations), 0)
    
    def test_empty_input(self):
        """Test the forward method with empty input."""
        # Call the forward method with empty input
        score = self.metric_module("", "Some prediction")
        
        # Check that the LM was called
        self.mock_lm.assert_called_once()
        
        # Check that a valid score is returned
        self.assertEqual(score, 0.75)
        
    def test_empty_prediction(self):
        """Test the forward method with empty prediction."""
        # Call the forward method with empty prediction
        score = self.metric_module("Some input", "")
        
        # Check that the LM was called
        self.mock_lm.assert_called_once()
        
        # Check that a valid score is returned
        self.assertEqual(score, 0.75)
        
    def test_none_input(self):
        """Test the forward method with None input."""
        # Call the forward method with None input
        score = self.metric_module(None, "Some prediction")
        
        # Check that a default score is returned
        self.assertEqual(score, 0.5)
        
    def test_none_prediction(self):
        """Test the forward method with None prediction."""
        # Call the forward method with None prediction
        score = self.metric_module("Some input", None)
        
        # Check that a default score is returned
        self.assertEqual(score, 0.5)
        
    def test_add_demonstration_with_user_score(self):
        """Test adding a demonstration with user_score instead of score."""
        # Initial state
        self.assertEqual(len(self.metric_module.demonstrations), 0)
        
        # Add a demonstration with user_score
        self.metric_module.add_demonstration(
            input="What is 1+1?",
            prediction="2",
            gold="2",
            user_score=1.0
        )
        
        # Check that the demonstration was added with the correct score
        self.assertEqual(len(self.metric_module.demonstrations), 1)
        self.assertEqual(self.metric_module.demonstrations[0]["score"], 1.0)
        
    def test_add_demonstration_missing_score(self):
        """Test adding a demonstration without a score."""
        # Initial state
        self.assertEqual(len(self.metric_module.demonstrations), 0)
        
        # Try to add a demonstration without a score
        with self.assertRaises(ValueError):
            self.metric_module.add_demonstration(
                input="What is 1+1?",
                prediction="2",
                gold="2"
            )
        
        # Check that no demonstration was added
        self.assertEqual(len(self.metric_module.demonstrations), 0)
        
    def test_parse_score_exception(self):
        """Test the _parse_score method when an exception occurs during parsing."""
        # Mock a scenario where an exception occurs during parsing
        with patch('builtins.float', side_effect=Exception("Parsing error")):
            score = self.metric_module._parse_score("0.75")
            # Check that the default score is returned
            self.assertEqual(score, 0.5)

if __name__ == '__main__':
    unittest.main()
