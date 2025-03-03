import sys
import os
import unittest
import tempfile
import shutil
import json
from datetime import datetime

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_learner.data_manager import MetricDataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a data manager with the test directory
        self.data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=self.test_dir
        )
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the data manager initializes correctly."""
        self.assertEqual(self.data_manager.metric_name, "test_metric")
        self.assertTrue(os.path.exists(self.data_manager.data_dir))
    
    def test_save_instance(self):
        """Test saving an instance."""
        # Save an instance
        filename = self.data_manager.save_instance(
            input="What is 2+2?",
            prediction="4",
            gold="4",
            score=0.9
        )
        
        # Check that the file exists
        self.assertTrue(os.path.exists(filename))
        
        # Check the contents of the file
        with open(filename, "r") as f:
            instance = json.load(f)
        
        self.assertEqual(instance["input"], "What is 2+2?")
        self.assertEqual(instance["prediction"], "4")
        self.assertEqual(instance["gold"], "4")
        self.assertEqual(instance["score"], 0.9)
        self.assertIsNone(instance["user_score"])
        self.assertIn("datetime", instance)
    
    def test_load_instances_empty(self):
        """Test loading instances when there are none."""
        instances = self.data_manager.load_instances()
        self.assertEqual(len(instances), 0)
    
    def test_load_instances(self):
        """Test loading instances after saving some."""
        # Save some instances
        self.data_manager.save_instance("Q1", "A1", score=0.1)
        self.data_manager.save_instance("Q2", "A2", score=0.2)
        self.data_manager.save_instance("Q3", "A3", score=0.3)
        
        # Load the instances
        instances = self.data_manager.load_instances()
        
        # Check that we got the right number of instances
        self.assertEqual(len(instances), 3)
        
        # Check that they're sorted by datetime
        self.assertLessEqual(instances[0]["datetime"], instances[1]["datetime"])
        self.assertLessEqual(instances[1]["datetime"], instances[2]["datetime"])
    
    def test_update_user_score(self):
        """Test updating a user score."""
        # Save an instance
        self.data_manager.save_instance("Q1", "A1", score=0.5)
        
        # Load the instance to get its datetime
        instances = self.data_manager.load_instances()
        self.assertEqual(len(instances), 1)
        
        # Update the user score
        success = self.data_manager.update_user_score(instances[0]["datetime"], 0.8)
        self.assertTrue(success)
        
        # Load the instance again to check the update
        instances = self.data_manager.load_instances()
        self.assertEqual(instances[0]["user_score"], 0.8)
    
    def test_update_nonexistent_instance(self):
        """Test updating a user score for a nonexistent instance."""
        # Try to update an instance that doesn't exist
        success = self.data_manager.update_user_score("nonexistent", 0.5)
        self.assertFalse(success)
    
    def test_get_labeled_dataset_empty(self):
        """Test getting a labeled dataset when there are no labeled instances."""
        # Save an instance without a user score
        self.data_manager.save_instance("Q1", "A1", score=0.5)
        
        # Get the labeled dataset
        dataset = self.data_manager.get_labeled_dataset()
        
        # Check that it's empty
        self.assertEqual(len(dataset), 0)
    
    def test_get_labeled_dataset(self):
        """Test getting a labeled dataset with labeled instances."""
        # Save an instance
        self.data_manager.save_instance("Q1", "A1", gold="A1", score=0.5)
        
        # Load the instance to get its datetime
        instances = self.data_manager.load_instances()
        
        # Update the user score
        self.data_manager.update_user_score(instances[0]["datetime"], 0.8)
        
        # Get the labeled dataset
        dataset = self.data_manager.get_labeled_dataset()
        
        # Check that we got one example
        self.assertEqual(len(dataset), 1)
        
        # Check the example's attributes
        example = dataset[0]
        self.assertEqual(example.input, "Q1")
        self.assertEqual(example.prediction, "A1")
        self.assertEqual(example.gold, "A1")
        self.assertEqual(example.user_score, 0.8)

if __name__ == '__main__':
    unittest.main()
