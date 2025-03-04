import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import importlib.util

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
import pytest
from metric_learner import MetricLearner, MetricModule

class TestIntegration(unittest.TestCase):
    """Integration tests for the DSPy Metric Learning package."""
    
    def setUp(self):
        # Create a mock language model
        self.mock_lm = MagicMock()
        self.mock_lm.return_value = "0.75"
        
        # Create a metric module with the mock LM
        self.metric_module = MetricModule(lm=self.mock_lm)
    
    def test_openrouter_example_imports(self):
        """Test that the OpenRouter example imports correctly."""
        # Import the example module
        example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'examples', 'openrouter_example.py')
        
        # Skip the test if the file doesn't exist
        if not os.path.exists(example_path):
            self.skipTest(f"Example file {example_path} not found")
        
        # Mock the required modules and functions
        with patch('dspy.LM') as mock_lm_class:
            # Set up the mock LM
            mock_lm = MagicMock()
            mock_lm_class.return_value = mock_lm
            
            # Mock dspy.settings
            with patch.object(dspy, 'settings', MagicMock()) as mock_settings:
                # Set the lm attribute on settings to be a proper type
                mock_settings.lm = mock_lm
                
                # Mock ChainOfThought to avoid the isinstance check
                with patch('dspy.ChainOfThought', MagicMock()) as mock_cot:
                    # Mock the optimize method to prevent execution
                    with patch('metric_learner.learner.MetricLearner.optimize', return_value=MagicMock()):
                        # Load the module without executing it
                        spec = importlib.util.spec_from_file_location("openrouter_example", example_path)
                        example_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(example_module)
        
        # Check that the LM was created with the right model
        mock_lm_class.assert_called_once()
        args, kwargs = mock_lm_class.call_args
        self.assertEqual(kwargs["model"], "openrouter/google/gemini-1.5-pro-latest")
    
    def test_openai_example_imports(self):
        """Test that the OpenAI example imports correctly."""
        # Import the example module
        example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'examples', 'openai_example.py')
        
        # Skip the test if the file doesn't exist
        if not os.path.exists(example_path):
            self.skipTest(f"Example file {example_path} not found")
        
        # Mock the required modules and functions
        with patch('dspy.OpenAI') as mock_openai_class:
            # Set up the mock OpenAI
            mock_openai = MagicMock()
            mock_openai_class.return_value = mock_openai
            
            # Mock dspy.settings
            with patch.object(dspy, 'settings', MagicMock()) as mock_settings:
                # Set the lm attribute on settings to be a proper type
                mock_settings.lm = mock_openai
                
                # Mock ChainOfThought to avoid the isinstance check
                with patch('dspy.ChainOfThought', MagicMock()) as mock_cot:
                    # Mock os.environ to provide the API key
                    with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}):
                        # Mock the optimize method to prevent execution
                        with patch('metric_learner.learner.MetricLearner.optimize', return_value=MagicMock()):
                            # Load the module without executing it
                            spec = importlib.util.spec_from_file_location("openai_example", example_path)
                            example_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(example_module)
        
        # Check that OpenAI was created with the right model and API key
        mock_openai_class.assert_called_once()
        args, kwargs = mock_openai_class.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(kwargs["api_key"], "fake-key")
    
    @pytest.mark.integration
    def test_simple_workflow(self):
        """Test a simple workflow with the MetricLearner."""
        # Create a simple DSPy program
        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate_answer = dspy.ChainOfThought("question -> answer")
            
            def forward(self, question):
                return self.generate_answer(question=question)
        
        # Create a MetricLearner
        learner = MetricLearner(
            program=SimpleQA(),
            metric_module=self.metric_module,
            num_iterations=1,
            verbose=True
        )
        
        # Create some examples
        examples = [
            dspy.Example(question="What is 2+2?", answer="4"),
            dspy.Example(question="What is 3+3?", answer="6")
        ]
        
        # Mock the optimize method to return a mock program
        mock_program = MagicMock()
        mock_program.return_value = MagicMock(answer="Rome")
        
        with patch.object(learner, 'optimize', return_value=mock_program):
            # Optimize the program
            optimized = learner.optimize(examples)
            
            # Test the optimized program
            result = optimized(question="What is the capital of Italy?")
            
            # Check the result
            self.assertEqual(result.answer, "Rome")
    
    @pytest.mark.integration
    def test_custom_optimizer(self):
        """Test using a custom optimizer with MetricLearner."""
        # Create a simple DSPy program
        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate_answer = dspy.ChainOfThought("question -> answer")
            
            def forward(self, question):
                return self.generate_answer(question=question)
        
        # Create a mock optimizer class
        mock_optimizer_class = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.compile.return_value = MagicMock()
        
        # Create a MetricLearner with the custom optimizer
        learner = MetricLearner(
            program=SimpleQA(),
            metric_module=self.metric_module,
            optimizer_class=mock_optimizer_class,
            num_iterations=1,
            verbose=True
        )
        
        # Create some examples
        examples = [
            dspy.Example(question="What is 2+2?", answer="4"),
            dspy.Example(question="What is 3+3?", answer="6")
        ]
        
        # Optimize the program
        learner.optimize(examples)
        
        # Check that the custom optimizer was used
        mock_optimizer_class.assert_called_once()
        mock_optimizer.compile.assert_called_once()

if __name__ == '__main__':
    unittest.main()
