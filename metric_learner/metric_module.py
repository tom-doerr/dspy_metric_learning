import dspy

class MetricModule(dspy.Module):
    """
    A DSPy module that uses a language model to rate predictions.
    
    This module takes an input, prediction, and optional gold standard,
    then uses a language model to generate a score between 0 and 1.
    """
    
    def __init__(self, lm, demonstrations=None, prompt_template=None):
        """
        Initialize the metric module.
        
        Args:
            lm: The language model to use for scoring
            demonstrations: Optional list of few-shot examples
            prompt_template: Optional custom prompt template
        """
        super().__init__()
        self.lm = lm
        self.demonstrations = demonstrations if demonstrations is not None else []
        
        if prompt_template is None:
            self.prompt_template = (
                "Rate the quality of the answer '{prediction}' for the question '{input}' "
                "on a scale from 0 to 1, where 0 is completely incorrect and 1 is perfect."
            )
        else:
            self.prompt_template = prompt_template

    def forward(self, input, prediction, gold=None):
        """
        Generate a score for a prediction.
        
        Args:
            input: The input text (e.g., a question)
            prediction: The prediction to score
            gold: Optional gold standard answer
            
        Returns:
            float: A score between 0 and 1
        """
        try:
            # Check for None inputs
            if input is None or prediction is None:
                return 0.5
                
            # Build prompt with demonstrations (if any)
            prompt = self._build_prompt(input, prediction, gold)
            
            # Query the language model
            response = self.lm(prompt)
            score = self._parse_score(response)
            return score
        except Exception as e:
            # If any error occurs, return a default score of 0.5
            return 0.5
    
    def _build_prompt(self, input, prediction, gold=None):
        """Build the prompt with optional demonstrations and gold standard."""
        prompt = ""
        
        # Add demonstrations if available
        if self.demonstrations:
            prompt += "Here are some examples of how to rate answers:\n\n"
            for demo in self.demonstrations:
                prompt += f"Question: {demo['input']}\n"
                prompt += f"Answer: {demo['prediction']}\n"
                if 'gold' in demo and demo['gold']:
                    prompt += f"Correct answer: {demo['gold']}\n"
                # Use 'score' or 'user_score' key, whichever is available
                score_key = 'score' if 'score' in demo else 'user_score'
                prompt += f"Rating: {demo[score_key]}\n\n"
            
            prompt += "Now, rate the following answer:\n\n"
        
        # Add the main prompt
        prompt += self.prompt_template.format(input=input, prediction=prediction)
        
        # Add gold standard if available
        if gold:
            prompt += f" The correct answer is '{gold}'."
        
        # Add final instruction
        prompt += "\n\nProvide only a number between 0 and 1 as your response."
        
        return prompt

    def _parse_score(self, response):
        """
        Extract a float score from the LM's response.
        
        Args:
            response: The LM's response string
            
        Returns:
            float: A score between 0 and 1
        """
        # Try to extract a float from the response
        try:
            # Clean the response and extract the first number
            cleaned = response.strip()
            
            # Look for a decimal number in the response (including negative numbers)
            import re
            match = re.search(r'(-?\d+\.\d+|-?\d+)', cleaned)
            if match:
                score = float(match.group(1))
                # Ensure the score is between 0 and 1
                if score < 0.0:
                    return 0.0
                elif score > 1.0:
                    return 1.0
                return score
            else:
                # If no number is found, return a default score
                return 0.5
        except Exception:
            # If parsing fails, return a default score
            return 0.5
            
    def get_learned_metric_fn(self):
        """
        Get a description of the learned metric function.
        
        This method returns a string representation of the current metric function,
        including the prompt template and any demonstrations.
        
        Returns:
            str: A description of the learned metric function
        """
        description = "Learned Metric Function:\n\n"
        
        # Add prompt template
        description += f"Prompt Template:\n{self.prompt_template}\n\n"
        
        # Add demonstrations if available
        if self.demonstrations:
            description += "Demonstrations:\n"
            for i, demo in enumerate(self.demonstrations, 1):
                description += f"Example {i}:\n"
                description += f"  Input: {demo['input']}\n"
                description += f"  Prediction: {demo['prediction']}\n"
                if 'gold' in demo and demo['gold']:
                    description += f"  Gold: {demo['gold']}\n"
                # Use 'score' or 'user_score' key, whichever is available
                score_key = 'score' if 'score' in demo else 'user_score'
                description += f"  Score: {demo[score_key]}\n\n"
        else:
            description += "No demonstrations available.\n"
            
        return description
        
    def add_demonstration(self, input, prediction, gold=None, score=None, user_score=None):
        """
        Add a demonstration to the metric module.
        
        Args:
            input: The input text (e.g., a question)
            prediction: The prediction to score
            gold: Optional gold standard answer
            score: The score for the demonstration (0-1)
            user_score: Alternative name for score parameter
            
        Returns:
            None
        """
        # Use user_score if score is None
        final_score = score if score is not None else user_score
        
        if final_score is None:
            raise ValueError("Either 'score' or 'user_score' must be provided")
            
        demo = {
            "input": input,
            "prediction": prediction,
            "score": final_score
        }
        
        if gold is not None:
            demo["gold"] = gold
            
        self.demonstrations.append(demo)
        
    def clear_demonstrations(self):
        """
        Clear all demonstrations from the metric module.
        
        Returns:
            None
        """
        self.demonstrations = []
