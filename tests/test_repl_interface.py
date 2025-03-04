import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import io

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_learner.repl_interface import label_instances
from metric_learner.data_manager import MetricDataManager

class TestREPLInterface(unittest.TestCase):
    def setUp(self):
        # Create a mock data manager
        self.data_manager = MagicMock(spec=MetricDataManager)
        self.data_manager.metric_name = "test_metric"
        
        # Set up instances with one unlabeled
        self.instances = [
            {"id": "1", "datetime": "2023-01-01", "input": "test input 1", "prediction": "test pred 1", "user_score": 0.8},
            {"id": "2", "datetime": "2023-01-02", "input": "test input 2", "prediction": "test pred 2", "user_score": None},
        ]
        
    def test_label_instances_no_unlabeled(self):
        """Test labeling when there are no unlabeled instances."""
        # Set up the data manager to return instances with all labeled
        self.data_manager.load_instances.return_value = [
            {"id": "1", "datetime": "2023-01-01", "input": "test input 1", "prediction": "test pred 1", "user_score": 0.8},
        ]
        
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            label_instances(self.data_manager)
            output = fake_out.getvalue()
            
        # Check that the correct message was printed
        self.assertIn("No unlabeled instances found", output)
        
        # Verify that update_user_score was not called
        self.data_manager.update_user_score.assert_not_called()
        
    @patch('builtins.input')
    def test_label_instances_with_score(self, mock_input):
        """Test labeling an instance with a valid score."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up the mock input to return a valid score and then 'q' to quit
        mock_input.side_effect = ["0.7", "q"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()):
            label_instances(self.data_manager)
        
        # Verify that update_user_score was called with the correct arguments
        self.data_manager.update_user_score.assert_called_once_with("2023-01-02", 0.7)
        
    @patch('builtins.input')
    def test_label_instances_with_invalid_score(self, mock_input):
        """Test labeling an instance with an invalid score."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up the mock input to return an invalid score, then a valid score, then 'q' to quit
        mock_input.side_effect = ["invalid", "1.5", "0.7", "q"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            label_instances(self.data_manager)
            output = fake_out.getvalue()
        
        # Verify that error messages were printed
        self.assertIn("Invalid input", output)
        self.assertIn("Score must be between 0 and 1", output)
        
        # Verify that update_user_score was called with the correct arguments
        self.data_manager.update_user_score.assert_called_once_with("2023-01-02", 0.7)
        
    @patch('builtins.input')
    def test_label_instances_skip(self, mock_input):
        """Test skipping an instance."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up the mock input to return 'skip' to skip and then 'exit' to quit
        mock_input.side_effect = ["skip", "exit"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()):
            label_instances(self.data_manager)
        
        # Verify that update_user_score was not called
        self.data_manager.update_user_score.assert_not_called()
        
    @patch('builtins.input')
    def test_label_instances_quit(self, mock_input):
        """Test quitting the labeling process."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up the mock input to return 'exit' to quit immediately
        mock_input.side_effect = ["exit"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()):
            label_instances(self.data_manager)
        
        # Verify that update_user_score was not called
        self.data_manager.update_user_score.assert_not_called()

    @patch('builtins.input')
    def test_label_instances_with_gold_and_score(self, mock_input):
        """Test displaying gold and model score."""
        # Set up the data manager to return one unlabeled instance with gold and score
        instance_with_gold_and_score = {
            "id": "3", 
            "datetime": "2023-01-03", 
            "input": "test input 3", 
            "prediction": "test pred 3", 
            "gold": "test gold 3",
            "score": 0.9,
            "user_score": None
        }
        self.data_manager.load_instances.return_value = [instance_with_gold_and_score]
        
        # Set up the mock input to return a valid score and then 'q' to quit
        mock_input.side_effect = ["0.7", "q"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            label_instances(self.data_manager)
            output = fake_out.getvalue()
        
        # Verify that gold and model score were displayed
        self.assertIn("Gold: test gold 3", output)
        self.assertIn("Model score: 0.9", output)
        
    @patch('builtins.input')
    def test_label_instances_help_command(self, mock_input):
        """Test the help command in the REPL."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up the mock input to return 'help', then a valid score, then 'q' to quit
        mock_input.side_effect = ["help", "0.7", "q"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            label_instances(self.data_manager)
            output = fake_out.getvalue()
        
        # Verify that help text was displayed
        self.assertIn("Enter a score between 0 and 1", output)
        self.assertIn("skip: Skip this instance", output)
        self.assertIn("exit: Exit the labeling session", output)
        self.assertIn("help: Show this help message", output)
        
    @patch('builtins.input')
    def test_label_instances_failed_save(self, mock_input):
        """Test handling of failed score saving."""
        # Set up the data manager to return one unlabeled instance
        self.data_manager.load_instances.return_value = self.instances
        
        # Set up update_user_score to return False (failed save)
        self.data_manager.update_user_score.return_value = False
        
        # Set up the mock input to return a valid score and then 'q' to quit
        mock_input.side_effect = ["0.7", "q"]
        
        # Run the function
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            label_instances(self.data_manager)
            output = fake_out.getvalue()
        
        # Verify that error message was displayed
        self.assertIn("Failed to save score", output)

if __name__ == '__main__':
    unittest.main()
