import json
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field


class EvaluatorResponse(BaseModel):
    result: str = Field()
    reasoning: str = Field()


class CommentorResponse(BaseModel):
    feedback: str = Field()


class HitRateCalculator:
    def __init__(self, api_client, gemini_client, max_attempts: int = 3):
        self.api_client = api_client
        self.gemini_client = gemini_client
        self.max_attempts = max_attempts
        self.results = {}
    
    async def evaluate_dataset(self, dataset: List[Dict[str, Any]], use_feedback: bool = False, 
                               response_type: str = "concise", user_id: str = "test_user") -> Dict[str, Any]:
        question_results = []
        hits_by_attempt = [0] * self.max_attempts
        total_questions = len(dataset)
        
        for i, question_data in enumerate(dataset):
            print(f"Processing question {i+1}/{total_questions}: {question_data['question'][:50]}...")
            
            result = await self.process_question(
                question_data=question_data,
                question_id=question_data.get("id", f"q{i}"),
                use_feedback=use_feedback,
                response_type=response_type,
                user_id=user_id
            )
            
            question_results.append(result)
            
            if result["hit_at"] > 0:
                hits_by_attempt[result["hit_at"] - 1] += 1
            
            await asyncio.sleep(0.5)
        
        hit_rates = {}
        for i in range(self.max_attempts):
            hit_rates[f"hit_rate@{i+1}"] = hits_by_attempt[i] / total_questions
        
        results_summary = {
            "total_questions": total_questions,
            "hit_rates": hit_rates,
            "hits_by_attempt": hits_by_attempt,
            "use_feedback": use_feedback,
            "question_results": question_results
        }
        
        result_key = f"feedback_{use_feedback}_{int(time.time())}"
        self.results[result_key] = results_summary
        
        return results_summary
    
    async def process_question(self, question_data: Dict[str, Any], question_id: Optional[str] = None,
                              use_feedback: bool = False, response_type: str = "concise", 
                              user_id: str = "test_user") -> Dict[str, Any]:
        question = question_data["question"]
        ground_truth = question_data["ground_truth"]
        
        if question_id is None:
            question_id = f"q_{hash(question) % 10000}"
        
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "attempts": [],
            "hit_at": 0
        }
        
        conversation_history = [
            {
                "role": "user",
                "content": "Hello, I'd like to ask some questions about diabetes."
            }
        ]
        
        for attempt in range(1, self.max_attempts + 1):
            if result["hit_at"] > 0:
                result["attempts"].append({
                    "attempt": attempt,
                    "skipped": True
                })
                continue
            
            current_message = {
                "role": "user",
                "content": question
            }
            
            if use_feedback and attempt > 1:
                prev_attempt = result["attempts"][-1]
                if "feedback" in prev_attempt and prev_attempt["feedback"]:
                    current_message["content"] = self._create_next_prompt(
                        question,
                        prev_attempt["response"],
                        prev_attempt["feedback"],
                        attempt
                    )
            
            if attempt > 1:
                conversation_history.append({
                    "role": "assistant",
                    "content": result["attempts"][-1]["response"]
                })
            
            conversation_history.append(current_message)
            
            response = self.api_client.get_answer(
                prompt=current_message["content"],
                conversation_history=conversation_history,
                response_type=response_type,
                user_id=user_id
            )
            
            is_hit, eval_result, reasoning = await self._evaluate_answer(
                question=question,
                ground_truth=ground_truth,
                response=response
            )
            
            attempt_result = {
                "attempt": attempt,
                "prompt": current_message["content"],
                "response": response,
                "is_hit": is_hit,
                "evaluation": eval_result,
                "reasoning": reasoning
            }
            
            if is_hit:
                result["hit_at"] = attempt
            
            if not is_hit and attempt < self.max_attempts and use_feedback:
                feedback = await self._generate_feedback(
                    question=question,
                    ground_truth=ground_truth,
                    response=response,
                    reasoning=reasoning
                )
                attempt_result["feedback"] = feedback
            
            result["attempts"].append(attempt_result)
        
        return result
    
    async def _evaluate_answer(self, question: str, ground_truth: Dict[str, Any], 
                              response: str) -> Tuple[bool, str, str]:
        prompt = self._create_evaluator_prompt(question, ground_truth, response)
        
        try:
            evaluation = await self.gemini_client.generate(
                prompt=prompt,
                format=EvaluatorResponse
            )
            
            if isinstance(evaluation, list) and len(evaluation) > 0:
                eval_result = evaluation[0]
            elif hasattr(evaluation, "result"):
                eval_result = evaluation
            elif isinstance(evaluation, dict) and "message" in evaluation:
                content = evaluation["message"]["content"]
                if hasattr(content, "result"):
                    eval_result = content
                else:
                    eval_result = EvaluatorResponse(**content)
            else:
                eval_result = evaluation
            
            is_hit = eval_result.result.lower() == "accurate"
            
            return is_hit, eval_result.result, eval_result.reasoning
            
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return self._fallback_evaluation(response, ground_truth)
    
    def _fallback_evaluation(self, response: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, str]:
        if "answer" in ground_truth:
            answer = ground_truth.get("answer", "").strip().lower()
            is_hit = answer in response.strip().lower()
            
            if is_hit:
                return True, "accurate", f"Response contains the correct answer '{answer}'."
            else:
                return False, "incorrect", f"Response does not contain the correct answer '{answer}'."
        
        elif "required_elements" in ground_truth:
            required_elements = ground_truth.get("required_elements", [])
            required_count = ground_truth.get("required_count", len(required_elements))
            
            matched_elements = []
            for element in required_elements:
                if element.lower() in response.lower():
                    matched_elements.append(element)
            
            is_hit = len(matched_elements) >= required_count
            
            if is_hit:
                return True, "accurate", f"Response contains {len(matched_elements)} of the required elements."
            else:
                return False, "missing", f"Response only contains {len(matched_elements)} of the {required_count} required elements."
        
        return False, "incorrect", "Unable to evaluate response against ground truth."
    
    def _create_evaluator_prompt(self, question: str, ground_truth: Dict[str, Any], response: str) -> str:
        ground_truth_text = ""
        
        if "answer" in ground_truth:
            ground_truth_text = ground_truth.get("answer", "")
        elif "required_elements" in ground_truth:
            elements = ground_truth.get("required_elements", [])
            ground_truth_text = ", ".join(elements)
        
        prompt = f"""
### Question: {question}
### True Answer: {ground_truth_text}
### Predicted Answer: {response}
### Task: Based on the question and the true answer, is the predicted answer accurate, incorrect, or missing? The answer must be one of them and is in one word. Then provide your reasoning.
        """
        
        return prompt.strip()
    
    async def _generate_feedback(self, question: str, ground_truth: Dict[str, Any], 
                                response: str, reasoning: str) -> str:
        ground_truth_text = ""
        if "answer" in ground_truth:
            ground_truth_text = ground_truth.get("answer", "")
        elif "required_elements" in ground_truth:
            elements = ground_truth.get("required_elements", [])
            ground_truth_text = ", ".join(elements)
        
        prompt = f"""
You are a helpful, pattern-following assistant.

Question: {question}
True Answer: {ground_truth_text}
Predicted Answer: {response}

Please point out the wrong parts of the predicted answer and provide helpful feedback to improve it.
Your feedback should:
1. Be constructive and clear
2. Point out what specific information is missing or incorrect
3. Guide them toward providing a more complete or accurate answer
4. Not be too leading (don't give away the complete answer)

Provide your feedback as a short paragraph (2-3 sentences).
        """
        
        try:
            feedback_response = await self.gemini_client.generate(
                prompt=prompt.strip(),
                format=CommentorResponse
            )
            
            if isinstance(feedback_response, list) and len(feedback_response) > 0:
                result = feedback_response[0]
            elif hasattr(feedback_response, "feedback"):
                result = feedback_response
            elif isinstance(feedback_response, dict) and "message" in feedback_response:
                content = feedback_response["message"]["content"]
                if hasattr(content, "feedback"):
                    result = content
                else:
                    result = CommentorResponse(**content)
            else:
                result = feedback_response
            
            return result.feedback
            
        except Exception as e:
            print(f"Error generating feedback: {str(e)}")
            return f"Your answer needs improvement. {reasoning} Please try again with a more specific answer."
    
    def _create_next_prompt(self, question: str, response: str, feedback: str, attempt: int) -> str:
        return f"{question}\n\nYour previous answer: \"{response}\"\n\nFeedback: {feedback}\n\nPlease provide an improved answer based on this feedback."
    
    def save_results(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    async def get_summary_stats(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No evaluation results available"}
        
        feedback_comparison = {
            "with_feedback": {"count": 0, "hit_rates": {}},
            "without_feedback": {"count": 0, "hit_rates": {}}
        }
        
        for key, result in self.results.items():
            category = "with_feedback" if result.get("use_feedback", False) else "without_feedback"
            feedback_comparison[category]["count"] += 1
            
            for hit_key, value in result.get("hit_rates", {}).items():
                if hit_key not in feedback_comparison[category]["hit_rates"]:
                    feedback_comparison[category]["hit_rates"][hit_key] = 0
                feedback_comparison[category]["hit_rates"][hit_key] += value
        
        for category, data in feedback_comparison.items():
            if data["count"] > 0:
                for hit_key in data["hit_rates"]:
                    data["hit_rates"][hit_key] /= data["count"]
        
        return {
            "total_evaluations": len(self.results),
            "feedback_comparison": feedback_comparison,
            "results": self.results
        }


def create_dataset(questions, ground_truths):
    dataset = []
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        dataset.append({
            "id": f"q{i}",
            "question": question,
            "ground_truth": ground_truth
        })
    return dataset


async def main():
    import os
    import sys
    sys.path.append(".")

    from utils import DiabetesKGAPIClient
    from app.backend.llm.gemini_client import GeminiClient
    from dotenv import load_dotenv
    
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
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
    
    questions = [
        "What is diabetes? a. Body produces too much glucose b. Body does not make or use insulin properly c. Joints are stiff and painful d. a and b",
        "What are the main symptoms of type 2 diabetes?"
    ]
    
    ground_truths = [
        {
            "answer": "d"
        },
        {
            "required_elements": [
                "increased thirst", 
                "frequent urination", 
                "fatigue", 
                "blurred vision", 
                "slow healing"
            ],
            "required_count": 3
        }
    ]
    
    dataset = create_dataset(questions, ground_truths)
    
    for use_feedback in [False, True]:
        feedback_status = "with" if use_feedback else "without"
        print(f"\nEvaluating {feedback_status} feedback...")
        
        results = await calculator.evaluate_dataset(
            dataset=dataset,
            use_feedback=use_feedback,
            response_type="concise",
            user_id="test_user"
        )
        
        print(f"Results summary:")
        for i in range(1, calculator.max_attempts + 1):
            print(f"  Hit@{i}: {results['hit_rates'].get(f'hit_rate@{i}', 0):.2f}")
    
    calculator.save_results("hit_rate_results.json")
    
    print("\nEvaluation complete. Results saved to hit_rate_results.json")


if __name__ == "__main__":
    asyncio.run(main())
