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
        # Build prompt with demonstrations (if any)
        prompt = self._build_prompt(input, prediction, gold)
        
        # Query the language model
        response = self.lm(prompt)
        score = self._parse_score(response)
        return score
    
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
                prompt += f"Rating: {demo['user_score']}\n\n"
            
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
                if score < 0.0 or score > 1.0:
                    raise ValueError(f"Score {score} is outside the valid range of 0-1")
                return score
            else:
                raise ValueError("No number found in response")
        except Exception as e:
            # If parsing fails, raise the error
            raise ValueError(f"Failed to parse score from response: {response}") from e
