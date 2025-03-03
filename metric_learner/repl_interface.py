def label_instances(data_manager):
    """
    Interactive REPL for labeling instances with user scores.
    
    Displays each unlabeled instance and prompts the user for a score.
    
    Args:
        data_manager: MetricDataManager instance
    """
    # Load instances
    instances = data_manager.load_instances()
    
    # Filter for unlabeled instances
    unlabeled = [i for i in instances if i.get("user_score") is None]
    
    if not unlabeled:
        print("No unlabeled instances found.")
        return
    
    print(f"Found {len(unlabeled)} unlabeled instances for metric '{data_manager.metric_name}'.")
    print("Enter a score between 0 and 1, or one of the following commands:")
    print("  skip: Skip this instance")
    print("  exit: Exit the labeling session")
    print("  help: Show this help message")
    print()
    
    for idx, instance in enumerate(unlabeled):
        print(f"\n--- Instance {idx+1}/{len(unlabeled)} ---")
        print(f"Input: {instance['input']}")
        print(f"Prediction: {instance['prediction']}")
        
        if instance.get("gold"):
            print(f"Gold: {instance['gold']}")
            
        if instance.get("score") is not None:
            print(f"Model score: {instance['score']}")
        
        while True:
            user_input = input("\nYour score (0-1, skip, exit, help): ").strip().lower()
            
            if user_input == "skip":
                print("Skipping to next instance.")
                break
                
            elif user_input == "exit":
                print("Exiting labeling session.")
                return
                
            elif user_input == "help":
                print("Enter a score between 0 and 1, or one of the following commands:")
                print("  skip: Skip this instance")
                print("  exit: Exit the labeling session")
                print("  help: Show this help message")
                continue
                
            try:
                user_score = float(user_input)
                if 0 <= user_score <= 1:
                    # Update the instance with the user score
                    success = data_manager.update_user_score(instance["datetime"], user_score)
                    if success:
                        print(f"Score {user_score} saved.")
                    else:
                        print("Failed to save score. Please try again.")
                    break
                else:
                    print("Score must be between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 1, or 'skip', 'exit', or 'help'.")
    
    print("\nLabeling session complete.")
    labeled_count = len([i for i in instances if i.get("user_score") is not None])
    print(f"You have labeled {labeled_count}/{len(instances)} instances for metric '{data_manager.metric_name}'.")
