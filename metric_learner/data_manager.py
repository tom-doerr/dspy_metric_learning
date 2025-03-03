import os
import json
from datetime import datetime
import dspy

class MetricDataManager:
    """
    Manages storage and retrieval of metric instances in a dotfile directory.
    
    Each instance contains input, prediction, gold standard (optional),
    model score, user score (optional), and timestamp.
    """
    
    def __init__(self, metric_name, data_dir=".metrics_data"):
        """
        Initialize the data manager.
        
        Args:
            metric_name: Name of the metric (used for directory naming)
            data_dir: Base directory for storing metric data
        """
        self.metric_name = metric_name
        self.data_dir = os.path.join(os.path.expanduser("~"), data_dir, metric_name)
        os.makedirs(self.data_dir, exist_ok=True)

    def save_instance(self, input, prediction, gold=None, score=None):
        """
        Save a new instance to the data directory.
        
        Args:
            input: The input text
            prediction: The prediction text
            gold: Optional gold standard answer
            score: Optional model-generated score
            
        Returns:
            str: Filename of the saved instance
        """
        # Create instance data
        instance = {
            "input": input,
            "prediction": prediction,
            "gold": gold,
            "score": score,
            "user_score": None,
            "datetime": datetime.now().isoformat()
        }
        
        # Generate filename based on timestamp
        timestamp = instance["datetime"].replace(":", "-").replace(".", "-")
        filename = os.path.join(self.data_dir, f"{timestamp}.json")
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(instance, f, indent=2)
            
        return filename

    def load_instances(self):
        """
        Load all instances from the data directory.
        
        Returns:
            list: Sorted list of instances by datetime
        """
        instances = []
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            return instances
            
        # Load each JSON file
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.data_dir, filename), "r") as f:
                        instance = json.load(f)
                        instances.append(instance)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        # Sort by datetime
        return sorted(instances, key=lambda x: x.get("datetime", ""))

    def update_user_score(self, datetime_str, user_score):
        """
        Update an instance with a user-provided score.
        
        Args:
            datetime_str: Datetime string of the instance
            user_score: User-provided score (float between 0 and 1)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        # Find the file with matching datetime
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        instance = json.load(f)
                    
                    # Check if this is the instance we're looking for
                    if instance.get("datetime") == datetime_str:
                        # Update the user score
                        instance["user_score"] = user_score
                        
                        # Save back to file
                        with open(filepath, "w") as f:
                            json.dump(instance, f, indent=2)
                            
                        return True
                except Exception as e:
                    print(f"Error updating {filename}: {e}")
                    
        return False
        
    def get_labeled_dataset(self):
        """
        Convert labeled instances to a DSPy dataset.
        
        Returns:
            list: List of dspy.Example objects
        """
        instances = self.load_instances()
        dataset = []
        
        for instance in instances:
            # Only include instances with user scores
            if instance.get("user_score") is not None:
                # Create a DSPy Example
                example = dspy.Example(
                    input=instance["input"],
                    prediction=instance["prediction"],
                    gold=instance.get("gold"),
                    user_score=instance["user_score"]
                ).with_inputs("input", "prediction", "gold")
                
                dataset.append(example)
                
        return dataset
