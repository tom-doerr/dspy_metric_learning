import sys
import os
import unittest
import pytest

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_learner import (
    MetricModule,
    MetricDataManager,
    optimize_metric_module,
    MetricEvaluator
)

class MockLM:
    """Mock language model for testing."""
    def __call__(self, prompt):
        return "0.75"

@pytest.mark.integration
class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests for the metric learning package."""
    
    def setUp(self):
        """Set up test environment."""
        self.lm = MockLM()
        self.metric_module = MetricModule(lm=self.lm)
        
        # Use a test-specific directory to avoid conflicts with user data
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        self.data_manager = MetricDataManager(
            metric_name="test_metric", 
            data_dir=test_data_dir
        )
        
        # Clean up any existing test data
        metric_dir = os.path.join(test_data_dir, "test_metric")
        if os.path.exists(metric_dir):
            import shutil
            shutil.rmtree(metric_dir)
            
        # Recreate the directory
        os.makedirs(metric_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        metric_dir = os.path.join(os.path.dirname(__file__), "test_data", "test_metric")
        if os.path.exists(metric_dir):
            import shutil
            shutil.rmtree(metric_dir)
    
    @pytest.mark.slow
    def test_full_workflow(self):
        """Test the complete metric learning workflow."""
        # 1. Create instances
        for i in range(3):
            self.data_manager.save_instance(
                input=f"Question {i}?",
                prediction=f"Answer {i}",
                gold=f"Gold {i}" if i % 2 == 0 else None
            )
        
        # 2. Label instances
        instances = self.data_manager.load_instances()
        for instance in instances:
            self.data_manager.update_user_score(instance["datetime"], 0.8)
        
        # 3. Optimize metric
        dataset = self.data_manager.get_labeled_dataset()
        optimized_module = optimize_metric_module(self.metric_module, dataset)
        
        # 4. Evaluate optimized metric
        evaluator = MetricEvaluator(optimized_module, self.data_manager)
        metrics = evaluator.evaluate()
        
        # 5. Verify results
        self.assertIn("mse", metrics)
        self.assertIn("correlation", metrics)
        self.assertEqual(metrics["num_examples"], 3)
    
    def test_basic_integration(self):
        """A faster integration test that doesn't use the slow marker."""
        # Create a single instance
        self.data_manager.save_instance(
            input="Test question?",
            prediction="Test answer",
            gold="Test gold"
        )
        
        # Score it
        instances = self.data_manager.load_instances()
        self.data_manager.update_user_score(instances[0]["datetime"], 0.7)
        
        # Get dataset
        dataset = self.data_manager.get_labeled_dataset()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].user_score, 0.7)

if __name__ == '__main__':
    unittest.main()
