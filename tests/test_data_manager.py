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
    
    def test_load_instances_with_corrupt_file(self):
        """Test loading instances when there's a corrupt JSON file."""
        # Create a corrupt JSON file
        corrupt_file = os.path.join(self.test_dir, "corrupt.json")
        with open(corrupt_file, "w") as f:
            f.write("This is not valid JSON")
        
        # Load instances
        instances = self.data_manager.load_instances()
        
        # Should still get an empty list (corrupt file is skipped)
        self.assertEqual(len(instances), 0)
    
    def test_update_user_score_with_corrupt_file(self):
        """Test updating a user score when there's a corrupt JSON file."""
        # Save an instance
        filename = self.data_manager.save_instance("Q1", "A1", score=0.5)
        
        # Load the instance to get its datetime
        instances = self.data_manager.load_instances()
        datetime_str = instances[0]["datetime"]
        
        # Make the file read-only to cause a write error
        os.chmod(filename, 0o444)
        
        try:
            # Try to update the user score
            success = self.data_manager.update_user_score(datetime_str, 0.8)
            
            # Should fail due to permissions
            self.assertFalse(success)
        finally:
            # Restore permissions for cleanup
            os.chmod(filename, 0o644)
            
    def test_load_instances_nonexistent_directory(self):
        """Test loading instances from a non-existent directory."""
        # Create a data manager with a non-existent directory
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=nonexistent_dir
        )
        
        # Load instances
        instances = data_manager.load_instances()
        
        # Should get an empty list
        self.assertEqual(len(instances), 0)
        
    def test_load_instances_with_invalid_json(self):
        """Test loading instances with invalid JSON content."""
        # Create a file with invalid JSON content
        invalid_file = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_file, "w") as f:
            f.write("This is not valid JSON")
            
        # Load instances
        instances = self.data_manager.load_instances()
        
        # Should get an empty list (invalid file is skipped with error message)
        self.assertEqual(len(instances), 0)

    def test_load_instances_directory_error(self):
        """Test loading instances when directory access fails."""
        # Create a non-existent directory that we'll try to access
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        
        # Create a data manager with the non-existent directory
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=nonexistent_dir
        )
        
        # Now let's make the directory exist but not be readable
        os.makedirs(nonexistent_dir, exist_ok=True)
        try:
            # Make the directory not readable
            os.chmod(nonexistent_dir, 0)
            
            # Try to load instances
            instances = data_manager.load_instances()
            
            # Should get an empty list
            self.assertEqual(len(instances), 0)
        finally:
            # Restore permissions for cleanup
            os.chmod(nonexistent_dir, 0o755)

    def test_load_instances_file_open_error(self):
        """Test loading instances when file open fails."""
        # Create a file
        filename = os.path.join(self.test_dir, "test.json")
        with open(filename, "w") as f:
            f.write('{"input": "test", "prediction": "test", "datetime": "2021-01-01"}')
            
        # Mock open to raise an exception
        original_open = open
        
        def mock_open(*args, **kwargs):
            if args[0] == os.path.join(self.test_dir, "test.json") and args[1] == "r":
                raise PermissionError("Permission denied")
            return original_open(*args, **kwargs)
        
        # Create a data manager
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=self.test_dir
        )
        
        # Mock open
        with unittest.mock.patch('builtins.open', mock_open):
            # Load instances
            instances = data_manager.load_instances()
            
            # Should get an empty list
            self.assertEqual(len(instances), 0)

    def test_load_instances_json_decode_error(self):
        """Test loading instances with JSON decode error."""
        # Create a file with invalid JSON
        filename = os.path.join(self.test_dir, "invalid.json")
        with open(filename, "w") as f:
            f.write("{invalid json")
            
        # Create a data manager
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=self.test_dir
        )
        
        # Load instances
        instances = data_manager.load_instances()
        
        # Should get an empty list
        self.assertEqual(len(instances), 0)

    def test_load_instances_with_specific_json_error(self):
        """Test loading instances with a specific JSON error."""
        # Create a file that will cause a specific JSON error
        filename = os.path.join(self.test_dir, "bad_json.json")
        with open(filename, "w") as f:
            # This will cause a JSONDecodeError when loaded
            f.write('{"key": bad_value}')
            
        # Create a data manager
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=self.test_dir
        )
        
        # Load instances (should skip the bad file)
        instances = data_manager.load_instances()
        
        # Should get an empty list
        self.assertEqual(len(instances), 0)

    def test_load_instances_with_exception_handling(self):
        """Test the exception handling in load_instances method."""
        # Create a file with invalid JSON content
        invalid_file = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_file, "w") as f:
            f.write('{invalid json')
            
        # Create a valid file
        valid_file = os.path.join(self.test_dir, "valid.json")
        with open(valid_file, "w") as f:
            json.dump({"input": "test1", "prediction": "test1", "datetime": "2021-01-01"}, f)
        
        # Create a data manager with the test directory
        data_manager = MetricDataManager(
            metric_name="test_metric",
            data_dir=self.test_dir
        )
        
        # Create another valid file directly in the data_manager's directory
        os.makedirs(data_manager.data_dir, exist_ok=True)
        direct_valid_file = os.path.join(data_manager.data_dir, "direct_valid.json")
        with open(direct_valid_file, "w") as f:
            json.dump({"input": "test2", "prediction": "test2", "datetime": "2021-01-02"}, f)
        
        # Create another invalid file directly in the data_manager's directory
        direct_invalid_file = os.path.join(data_manager.data_dir, "direct_invalid.json")
        with open(direct_invalid_file, "w") as f:
            f.write('{invalid json')
            
        # Load instances
        instances = data_manager.load_instances()
        
        # Should get only the valid instance
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["input"], "test2")

if __name__ == '__main__':
    unittest.main()
