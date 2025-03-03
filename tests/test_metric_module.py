import sys
import os
import unittest
from unittest.mock import MagicMock

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
        """Test that the metric module initializes correctly."""
        self.assertEqual(self.metric_module.lm, self.mock_lm)
        self.assertEqual(len(self.metric_module.demonstrations), 0)
        self.assertIn("Rate the quality", self.metric_module.prompt_template)
    
    def test_custom_prompt_template(self):
        """Test that a custom prompt template is used when provided."""
        custom_template = "Custom prompt: {input} -> {prediction}"
        custom_module = MetricModule(lm=self.mock_lm, prompt_template=custom_template)
        self.assertEqual(custom_module.prompt_template, custom_template)
    
    def test_forward_basic(self):
        """Test the forward method with basic inputs."""
        score = self.metric_module("What is 2+2?", "4")
        
        # Check that the LM was called
        self.mock_lm.assert_called_once()
        
        # Check that the score is correct
        self.assertEqual(score, 0.75)
    
    def test_forward_with_gold(self):
        """Test the forward method with a gold standard."""
        score = self.metric_module("What is 2+2?", "4", gold="4")
        
        # Check that the LM was called with a prompt containing the gold standard
        args, _ = self.mock_lm.call_args
        prompt = args[0]
        self.assertIn("correct answer", prompt)
        self.assertIn("4", prompt)
    
    def test_forward_with_demonstrations(self):
        """Test the forward method with demonstrations."""
        # Create a module with demonstrations
        demos = [
            {"input": "What is 1+1?", "prediction": "2", "user_score": 1.0},
            {"input": "What is 1+2?", "prediction": "4", "user_score": 0.0}
        ]
        demo_module = MetricModule(lm=self.mock_lm, demonstrations=demos)
        
        # Call the forward method
        score = demo_module("What is 2+2?", "4")
        
        # Check that the LM was called with a prompt containing the demonstrations
        args, _ = self.mock_lm.call_args
        prompt = args[0]
        self.assertIn("What is 1+1?", prompt)
        self.assertIn("What is 1+2?", prompt)
    
    def test_parse_score_valid(self):
        """Test parsing a valid score."""
        self.mock_lm.return_value = "0.85"
        score = self.metric_module("What is 2+2?", "4")
        self.assertEqual(score, 0.85)
    
    def test_parse_score_invalid(self):
        """Test parsing an invalid score raises an error."""
        self.mock_lm.return_value = "Not a number"
        
        with self.assertRaises(ValueError):
            self.metric_module("What is 2+2?", "4")
    
    def test_parse_score_out_of_range_high(self):
        """Test parsing a score greater than 1."""
        self.mock_lm.return_value = "1.5"
        with self.assertRaises(ValueError):
            self.metric_module("What is 2+2?", "4")
    
    def test_parse_score_out_of_range_low(self):
        """Test parsing a score less than 0."""
        self.mock_lm.return_value = "-0.5"
        with self.assertRaises(ValueError):
            self.metric_module("What is 2+2?", "4")

if __name__ == '__main__':
    unittest.main()
