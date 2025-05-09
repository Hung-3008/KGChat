import json
import os
import asyncio
import time
import sys
sys.path.append(".")
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from utils import DiabetesKGAPIClient, GeminiHitChecker
from app.backend.llm.gemini_client import GeminiClient


class FeedbackResponse(BaseModel):
    """Pydantic model for structured feedback response."""
    feedback: str = Field(
        description="Helpful feedback to improve the answer"
    )


class FeedbackType(Enum):
    NONE = "none"
    SIMPLE = "simple"
    CORRECTIVE = "corrective"


class HitRateCalculator:
    def __init__(self, api_client: DiabetesKGAPIClient, gemini_client: GeminiClient, max_attempts: int = 3):
        self.api_client = api_client
        self.gemini_client = gemini_client
        self.hit_checker = GeminiHitChecker(gemini_client)
        self.max_attempts = max_attempts
        self.results = {}
        self.current_question = ""
    
    async def evaluate_dataset(self, dataset: List[Dict[str, Any]], feedback_type: FeedbackType = FeedbackType.NONE, response_type: str = "concise", user_id: str = "test_user") -> Dict[str, Any]:
        all_results = []
        hits_by_attempt = [0] * self.max_attempts
        total_questions = len(dataset)
        
        for i, question_data in enumerate(dataset):
            print(f"Processing question {i+1}/{total_questions}: {question_data['question'][:50]}...")
            
            result = await self.evaluate_question(
                question=question_data["question"],
                ground_truth=question_data["ground_truth"],
                question_id=question_data.get("id", f"q{i}"),
                feedback_type=feedback_type,
                response_type=response_type,
                user_id=user_id
            )
            all_results.append(result)
            if result["hit_at"] > 0:
                hits_by_attempt[result["hit_at"] - 1] += 1
            await asyncio.sleep(0.5)
        
        hit_rates = {}
        cumulative_hits = 0
        
        for i in range(self.max_attempts):
            hit_rates[f"hit_rate@{i+1}"] = hits_by_attempt[i] / total_questions
            cumulative_hits += hits_by_attempt[i]
            hit_rates[f"cumulative_hit_rate@{i+1}"] = cumulative_hits / total_questions
        
        results_summary = {
            "total_questions": total_questions,
            "hit_rates": hit_rates,
            "hits_by_attempt": hits_by_attempt,
            "feedback_type": feedback_type.value,
            "detailed_results": all_results
        }
        
        result_key = f"{feedback_type.value}_{int(time.time())}"
        self.results[result_key] = results_summary
        
        return results_summary
    
    async def evaluate_question(self, question: str, ground_truth: Dict[str, Any], question_id: Optional[str] = None, feedback_type: FeedbackType = FeedbackType.NONE, response_type: str = "concise", user_id: str = "test_user") -> Dict[str, Any]:
        self.current_question = question
        if question_id is None:
            question_id = f"q_{hash(question) % 10000}"
        
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "iterations": [],
            "hit_at": 0,
            "feedback_type": feedback_type.value
        }
        
        conversation_history = [
            {
                "role": "user",
                "content": "Hello, I'd like to ask some questions about diabetes."
            }
        ]
        
        for attempt in range(1, self.max_attempts + 1):
            if result["hit_at"] > 0:
                result["iterations"].append({
                    "attempt": attempt,
                    "skipped": True
                })
                continue
            
            current_message = {
                "role": "user",
                "content": question
            }
            
            if attempt > 1:
                prev_iteration = result["iterations"][-1]
                if "feedback_provided" in prev_iteration and prev_iteration["feedback_provided"]:
                    current_message["content"] = self._create_next_prompt(
                        question, 
                        prev_iteration["response"], 
                        prev_iteration["feedback_provided"], 
                        attempt
                    )
            
            if attempt > 1:
                conversation_history.append({
                    "role": "assistant",
                    "content": result["iterations"][-1]["response"]
                })
            
            conversation_history.append(current_message)
            
            response = self.api_client.get_answer(
                prompt=current_message["content"],
                conversation_history=conversation_history,
                response_type=response_type,
                user_id=user_id
            )
            
            is_hit, reasoning = await self._check_hit(response, ground_truth)
            
            iteration_result = {
                "attempt": attempt,
                "prompt": current_message["content"],
                "response": response,
                "is_hit": is_hit,
                "reasoning": reasoning
            }
            
            if is_hit:
                result["hit_at"] = attempt
            
            feedback = None
            if not is_hit and attempt < self.max_attempts:
                feedback = await self._generate_feedback(
                    question=question,
                    response=response,
                    reasoning=reasoning,
                    attempt=attempt,
                    feedback_type=feedback_type
                )
                iteration_result["feedback_provided"] = feedback
            
            result["iterations"].append(iteration_result)
        
        return result
    
    async def _check_hit(self, response: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a response satisfies the ground truth requirements using Gemini.
        
        Args:
            response: The API's response text
            ground_truth: Ground truth requirements to check against
            
        Returns:
            Tuple of (is_hit, reasoning)
        """
        return await self.hit_checker.check_hit(
            question=self.current_question,
            expected_answer=ground_truth,
            actual_response=response
        )
    
    async def _generate_feedback(self, question: str, response: str, reasoning: str, attempt: int, feedback_type: FeedbackType) -> Optional[str]:
        """
        Generate feedback based on the response and reasoning.
        
        Args:
            question: Original question
            response: Current response
            reasoning: Reasoning string explaining why the answer wasn't a hit
            attempt: Current attempt number
            feedback_type: Type of feedback to generate
            
        Returns:
            Feedback string or None
        """
        if feedback_type == FeedbackType.NONE:
            return None
        if feedback_type == FeedbackType.SIMPLE:
            return "Please try again with a more specific answer."
        if feedback_type == FeedbackType.CORRECTIVE:
            try:
                prompt = f"""
You are an expert providing feedback on an answer to a question.

QUESTION: {question}

CURRENT ANSWER: {response}

EVALUATION: {reasoning}

TASK:
Create helpful, specific feedback that will guide the responder to improve their answer.
The feedback should:
1. Be constructive and clear
2. Point out what specific information is missing or incorrect
3. Guide them toward providing a more complete or accurate answer
4. Not be too leading (don't give away the complete answer)

Provide your feedback as a short paragraph (2-3 sentences).
                """
                feedback_response = await self.gemini_client.generate(
                    prompt=prompt,
                    format=FeedbackResponse
                )
                
                # Extract feedback based on response format
                if isinstance(feedback_response, list) and len(feedback_response) > 0:
                    if hasattr(feedback_response[0], "feedback"):
                        feedback = feedback_response[0].feedback
                    else:
                        feedback = feedback_response[0].get("feedback", "")
                elif hasattr(feedback_response, "feedback"):
                    feedback = feedback_response.feedback
                elif isinstance(feedback_response, dict) and "message" in feedback_response:
                    content = feedback_response["message"]["content"]
                    if hasattr(content, "feedback"):
                        feedback = content.feedback
                    else:
                        feedback = content.get("feedback", "")
                else:
                    feedback = getattr(feedback_response, "feedback", "") or feedback_response.get("feedback", "")
                
                if feedback:
                    return feedback
            except Exception as e:
                print(f"Error generating corrective feedback with Gemini: {str(e)}")
        
        # Fallback to using the reasoning itself
        return f"Your previous answer needs improvement. {reasoning} Please try again."
    
    def _create_next_prompt(self, question: str, response: str, feedback: str, attempt: int) -> str:
        if feedback is None:
            return question
        return f"{question}\n\nYour previous answer: \"{response}\"\n\nFeedback: {feedback}"
    
    async def get_summary_stats(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No evaluation results available"}
        feedback_comparison = await self._compare_feedback_types()
        return {
            "total_evaluations": len(self.results),
            "feedback_comparison": feedback_comparison,
            "results_by_type": self.results
        }
    
    async def _compare_feedback_types(self) -> Dict[str, Any]:
        feedback_types = {}
        for key, result in self.results.items():
            feedback_type = result["feedback_type"]
            if feedback_type not in feedback_types:
                feedback_types[feedback_type] = []
            feedback_types[feedback_type].append(result)
        comparison = {}
        for feedback_type, results in feedback_types.items():
            if not results:
                continue
            avg_hit_rates = {}
            for i in range(1, self.max_attempts + 1):
                hit_key = f"hit_rate@{i}"
                cum_key = f"cumulative_hit_rate@{i}"
                avg_hit_rates[hit_key] = sum(r["hit_rates"].get(hit_key, 0) for r in results) / len(results)
                avg_hit_rates[cum_key] = sum(r["hit_rates"].get(cum_key, 0) for r in results) / len(results)
            comparison[feedback_type] = {
                "runs": len(results),
                "average_hit_rates": avg_hit_rates
            }
        return comparison
    
    def save_results(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    async def visualize_results(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No evaluation results available"}
        visualization_data = {
            "hit_rates_by_attempt": {},
            "cumulative_hit_rates": {},
            "feedback_comparison": []
        }
        feedback_comparison = await self._compare_feedback_types()
        for feedback_type, data in feedback_comparison.items():
            for i in range(1, self.max_attempts + 1):
                hit_key = f"hit_rate@{i}"
                cum_key = f"cumulative_hit_rate@{i}"
                if i not in visualization_data["hit_rates_by_attempt"]:
                    visualization_data["hit_rates_by_attempt"][i] = {}
                if i not in visualization_data["cumulative_hit_rates"]:
                    visualization_data["cumulative_hit_rates"][i] = {}
                visualization_data["hit_rates_by_attempt"][i][feedback_type] = data["average_hit_rates"][hit_key]
                visualization_data["cumulative_hit_rates"][i][feedback_type] = data["average_hit_rates"][cum_key]
            visualization_data["feedback_comparison"].append({
                "feedback_type": feedback_type,
                "hit_rate@1": data["average_hit_rates"]["hit_rate@1"],
                "cumulative_hit_rate@3": data["average_hit_rates"][f"cumulative_hit_rate@{self.max_attempts}"]
            })
        return visualization_data


async def main():
    import os 
    from dotenv import load_dotenv
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY_1")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    api_client = DiabetesKGAPIClient(
        api_url="http://localhost:8000",
        timeout=120
    )
    
    gemini_client = GeminiClient(
        api_key=gemini_api_key,
        model_name="gemini-2.0-flash"
    )
    
    calculator = HitRateCalculator(
        api_client=api_client,
        gemini_client=gemini_client,
        max_attempts=3
    )
    

    mc_dataset = [
        {
            "id": "mc1",
            "question": "What is diabetes? a. Body produces too much glucose b. Body does not make or use insulin properly c. Joints are stiff and painful d. a and b",
            "ground_truth": {
                "answer": "d",
                "exact_match": False
            }
        }
    ]
    
    for feedback_type in [FeedbackType.NONE, FeedbackType.SIMPLE, FeedbackType.CORRECTIVE]:
        results = await calculator.evaluate_dataset(
            mc_dataset, 
            feedback_type=feedback_type,
            response_type="concise",
            user_id="test_user"
        )
        print(f"Results with {feedback_type.value} feedback:")
        print(f"Hit@1: {results['hit_rates']['hit_rate@1']:.2f}")
        print(f"Hit@2: {results['hit_rates']['cumulative_hit_rate@2']:.2f}")
        print(f"Hit@3: {results['hit_rates']['cumulative_hit_rate@3']:.2f}")
    
    summary = await calculator.get_summary_stats()
    for feedback_type, data in summary["feedback_comparison"].items():
        print(f"\n{feedback_type.upper()}:")
        print(f"  Hit@1: {data['average_hit_rates']['hit_rate@1']:.2f}")
        print(f"  Hit@3 (cumulative): {data['average_hit_rates']['cumulative_hit_rate@3']:.2f}")
    
    calculator.save_results("hit_rate_evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main())