import requests
from typing import Dict, Any, Tuple, List, Optional
from app.backend.llm.gemini_client import GeminiClient
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional, Union

class HitEvaluation(BaseModel):
    """Pydantic model for structured hit evaluation results."""
    is_hit: bool = Field(
        description="Whether the response correctly answers the question"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Explanation of why the answer is or is not a hit"
    )



class DiabetesKGAPIClient:
    """Client for interacting with the Diabetes Knowledge Graph API."""
    
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 120):
        """
        Initialize the API client.
        
        Args:
            api_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
    
    def get_answer(self, prompt: str, conversation_history: Optional[List[Dict]] = None, 
                 response_type: str = "concise", user_id: str = "test_user") -> str:
        """
        Get an answer from the API for a given prompt.
        
        Args:
            prompt: The prompt to send to the API
            conversation_history: Optional conversation history
            response_type: Type of response ("concise" or "detailed")
            user_id: Optional user identifier
            
        Returns:
            Response string from the API
        """
        if conversation_history is None:
            conversation_history = [
                {
                    "role": "user",
                    "content": "Hello, I'd like to ask some questions about diabetes."
                }
            ]
        
        url = f"{self.api_url}/api/query"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": prompt,
            "conversation_history": conversation_history,
            "response_type": response_type,
            "user_id": user_id
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Return just the response text
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling API: {str(e)}")
            return f"ERROR: {str(e)}"




class GeminiHitChecker:
    """Class to check if an answer is a hit using your existing GeminiClient."""
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the hit checker with your GeminiClient.
        
        Args:
            gemini_client: Instance of your GeminiClient
        """
        self.gemini_client = gemini_client
    
    async def check_hit(self, question: str, expected_answer: Dict[str, Any], 
                       actual_response: str) -> Tuple[bool, str]:
        """
        Check if an answer is a hit using GeminiClient with structured output.
        
        Args:
            question: Original question
            expected_answer: Ground truth requirements
            actual_response: Response to evaluate
            
        Returns:
            Tuple of (is_hit, reasoning)
        """
        # Create the evaluation prompt
        prompt = self._create_hit_checker_prompt(question, expected_answer, actual_response)
        
        try:
            # Use your client's generate method with Pydantic model format
            response = await self.gemini_client.generate(
                prompt=prompt,
                format=HitEvaluation
            )
            
            # Parse the response based on format
            if isinstance(response, list) and len(response) > 0:
                result = response[0]
            elif hasattr(response, "is_hit"):  # Direct Pydantic model
                result = response
            elif isinstance(response, dict) and "message" in response:
                # If client returns dict with message
                content = response["message"]["content"]
                if hasattr(content, "is_hit"):
                    result = content
                else:
                    # Try to convert dict to Pydantic model
                    result = HitEvaluation(**content)
            else:
                # Fallback - try to convert response to HitEvaluation
                result = response
            
            # Extract the results
            is_hit = result.is_hit
            confidence = result.confidence
            reasoning = result.reasoning
            
            # Log detailed evaluation
            print(f"Hit evaluation for question: '{question[:50]}...'")
            print(f"Result: {'HIT' if is_hit else 'MISS'} (confidence: {confidence:.2f})")
            print(f"Reasoning: {reasoning[:100]}...")
            
            return is_hit, reasoning
            
        except Exception as e:
            print(f"Error using Gemini for hit checking: {str(e)}")
            
            # Fallback to basic checking if Gemini fails
            is_hit = self._fallback_hit_check(actual_response, expected_answer)
            fallback_reasoning = "Fallback evaluation used due to API error. "
            
            if expected_answer.get("exact_match", False):
                correct_answer = expected_answer.get("answer", "").strip().lower()
                if is_hit:
                    fallback_reasoning += f"Found the correct answer '{correct_answer}' in the response."
                else:
                    fallback_reasoning += f"Could not find the correct answer '{correct_answer}' in the response."
            else:
                elements = expected_answer.get("required_elements", [])
                fallback_reasoning += f"Checked for {len(elements)} required elements using basic string matching."
                
            return is_hit, fallback_reasoning
    
    def _create_hit_checker_prompt(self, question: str, expected_answer: Dict[str, Any], 
                                  actual_response: str) -> str:

        if expected_answer.get("exact_match", False):
            expected_section = f"""
Expected Answer: {expected_answer.get('answer', '')}
This is a multiple-choice question that requires an exact match.
            """
        else:
            elements_list = "\n".join([f"- {elem}" for elem in expected_answer.get("required_elements", [])])
            required_count = expected_answer.get("required_count", len(expected_answer.get("required_elements", [])))
            
            expected_section = f"""
Expected Elements:
{elements_list}

Required Count: At least {required_count} of these elements must be present for a correct answer.
            """
        
        # Construct the full prompt
        prompt = f"""
You are an expert evaluator determining if an answer correctly addresses a question.

QUESTION:
{question}

ACTUAL RESPONSE:
{actual_response}

EVALUATION CRITERIA:
{expected_section}

TASK:
Evaluate if the actual response correctly answers the question based on the evaluation criteria.

For an exact match question: Check if the response clearly indicates the correct option.
For an elements-based question: Check if the required number of elements are covered in the response.

Consider the following:
1. The response might paraphrase or use synonyms rather than exact wording
2. The response might include additional information beyond what's required
3. The answer may include explanations or context around the key points
4. For multiple choice, the answer might specify either the letter (e.g., "A") or the full option text

Analyze the response carefully and determine if it meets the criteria.
Provide thorough reasoning explaining why the answer does or does not meet the criteria.
        """
        
        return prompt.strip()
    
    def _fallback_hit_check(self, response: str, expected_answer: Dict[str, Any]) -> bool:

        if expected_answer.get("exact_match", False):
            correct_answer = expected_answer.get("answer", "").strip().lower()
            return correct_answer in response.strip().lower()
        else:
            elements = expected_answer.get("required_elements", [])
            required_count = expected_answer.get("required_count", len(elements))
            
            matched = sum(1 for element in elements if element.lower() in response.lower())
            return matched >= required_count
