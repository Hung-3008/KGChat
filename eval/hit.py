import json
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sys

# Add backend path to system path
sys.path.append("./app")
from backend.llm.gemini_client import GeminiClient
from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.llm.ollama_client import OllamaClient
from qdrant_client import QdrantClient
from backend.core.retrieval.kg_query_processor import run_query


class EvaluatorResponse(BaseModel):
    result: str = Field()
    reasoning: str = Field()


class CommentorResponse(BaseModel):
    feedback: str = Field()


class GeminiKeyManager:
    def __init__(
        self,
        current_key_index: int = 1,
        max_keys: int = 6,
        key_pattern: str = "GEMINI_API_KEY_{}"
    ):
        self.current_key_index = current_key_index
        self.max_keys = max_keys
        self.key_pattern = key_pattern
    
    def get_current_key(self) -> str:
        key_name = self.key_pattern.format(self.current_key_index)
        api_key = os.getenv(key_name)
        
        if not api_key:
            print(f"API key {key_name} not found in environment variables")
            return None
            
        return api_key
    
    def rotate_key(self) -> str:
        for _ in range(self.max_keys):
            self.current_key_index = (self.current_key_index % self.max_keys) + 1
            
            key_name = self.key_pattern.format(self.current_key_index)
            api_key = os.getenv(key_name)
            
            if api_key:
                print(f"Rotated to API key {key_name}")
                return api_key
        
        print("No valid API keys found after trying all options")
        return None


class BackendClientManager:
    """Manager for all backend clients"""
    
    def __init__(self, gemini_key_manager):
        self.gemini_key_manager = gemini_key_manager
        self.clients = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all backend clients"""
        if self.initialized:
            return self.clients
            
        load_dotenv()
        
        # Initialize Neo4j client
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Initialize Vector DB client
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        vector_db_client = VectorDBClient(
            host=qdrant_host,
            port=qdrant_port
        )
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            host=qdrant_host, 
            port=qdrant_port
        )
        
        # Initialize Ollama client
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        
        ollama_client = OllamaClient(
            host=ollama_host,
            embedding_model=embedding_model
        )
        
        # Initialize Gemini client with key manager
        gemini_api_key = self.gemini_key_manager.get_current_key()
        gemini_client = GeminiClient(
            api_key=gemini_api_key, 
            model_name="gemini-2.0-flash"
        )
        
        # Verify connectivity
        if not await neo4j_client.verify_connectivity():
            print("Failed to connect to Neo4j database")
            raise Exception("Neo4j connection failed")
            
        print("All backend clients initialized successfully")
        
        self.clients = {
            "neo4j_client": neo4j_client,
            "vector_db_client": vector_db_client,
            "ollama_client": ollama_client,
            "gemini_client": gemini_client,
            "qdrant_client": qdrant_client
        }
        
        self.initialized = True
        return self.clients
    
    def rotate_gemini_key(self):
        """Rotate Gemini API key when needed"""
        new_key = self.gemini_key_manager.rotate_key()
        if new_key:
            new_gemini_client = GeminiClient(
                api_key=new_key, 
                model_name="gemini-2.0-flash"
            )
            self.clients["gemini_client"] = new_gemini_client
            return new_gemini_client
        return None
    
    async def close(self):
        """Close all client connections"""
        if self.clients and "neo4j_client" in self.clients:
            await self.clients["neo4j_client"].close()
        self.initialized = False


class BackendAPIClient:
    """Adapter to mimic API client behavior but using backend services directly"""
    
    def __init__(self, client_manager: BackendClientManager, max_retries: int = 3):
        self.client_manager = client_manager
        self.max_retries = max_retries
    
    async def get_answer(self, prompt: str, conversation_history: List[Dict[str, str]] = None,
                        response_type: str = "concise", user_id: str = "test_user") -> str:
        """
        Get answer using backend services directly instead of HTTP API
        """
        if conversation_history is None:
            conversation_history = []
        
        clients = await self.client_manager.initialize()
        
        for attempt in range(self.max_retries):
            try:
                # Use the backend kg_query_processor directly
                result = await run_query(
                    query=prompt,
                    conversation_history=conversation_history,
                    clients=clients,
                    grounding=False,  # Set to True if you want web grounding
                    language="English"
                )
                
                if result and "response" in result:
                    return result["response"]
                else:
                    print(f"No response received on attempt {attempt + 1}")
                    
            except Exception as e:
                error_str = str(e).lower()
                print(f"Error on attempt {attempt + 1}/{self.max_retries}: {error_str}")
                
                # Check if it's a rate limit or quota error
                if any(keyword in error_str for keyword in ["resource_exhausted", "rate limit", "429", "quota"]):
                    # Try to rotate the Gemini API key
                    new_client = self.client_manager.rotate_gemini_key()
                    if new_client:
                        clients["gemini_client"] = new_client
                        print(f"Rotated to new Gemini API key (index: {self.client_manager.gemini_key_manager.current_key_index})")
                        await asyncio.sleep(2)  # Wait before retry
                    else:
                        print("No more valid Gemini API keys available")
                        if attempt == self.max_retries - 1:
                            break
                else:
                    # For other errors, wait a bit and retry
                    await asyncio.sleep(1)
                    
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2)  # Wait between retries
        
        print(f"Failed to get response after {self.max_retries} attempts")
        return "Error: Failed to get response from backend services"


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
            
            # Use the backend API client instead of HTTP API
            response = await self.api_client.get_answer(
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    load_dotenv()
    
    # Initialize key manager and client manager
    key_manager = GeminiKeyManager(current_key_index=1, max_keys=6)
    client_manager = BackendClientManager(key_manager)
    
    try:
        # Initialize backend API client
        api_client = BackendAPIClient(client_manager, max_retries=3)
        
        # Initialize separate Gemini client for evaluation
        gemini_api_key = key_manager.get_current_key()
        if not gemini_api_key:
            print("No valid Gemini API key found")
            return
            
        gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model_name="gemini-2.0-flash"
        )
        
        calculator = HitRateCalculator(
            api_client=api_client,
            gemini_client=gemini_client,
            max_attempts=3
        )
        
        # Define test questions and ground truths
        questions = [
            "What is diabetes? a. Body produces too much glucose b. Body does not make or use insulin properly c. Joints are stiff and painful d. a and b",
            "What are the main symptoms of type 2 diabetes?",
            "What are the complications of uncontrolled diabetes?",
            "How is diabetes diagnosed?",
            "What lifestyle changes can help manage diabetes?"
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
            },
            {
                "required_elements": [
                    "heart disease",
                    "kidney damage", 
                    "nerve damage",
                    "eye problems",
                    "foot problems"
                ],
                "required_count": 3
            },
            {
                "required_elements": [
                    "fasting glucose",
                    "A1C test",
                    "glucose tolerance test",
                    "random glucose"
                ],
                "required_count": 2
            },
            {
                "required_elements": [
                    "healthy diet",
                    "regular exercise",
                    "weight management",
                    "blood sugar monitoring"
                ],
                "required_count": 3
            }
        ]
        
        dataset = create_dataset(questions, ground_truths)
        
        # Create output directory
        output_dir = f"eval/hit_rate_results/{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate with and without feedback
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
                hit_rate = results['hit_rates'].get(f'hit_rate@{i}', 0)
                print(f"  Hit@{i}: {hit_rate:.2f}")
        
        # Save results
        results_file = f"{output_dir}/hit_rate_results.json"
        calculator.save_results(results_file)
        
        # Get and save summary statistics
        summary_stats = await calculator.get_summary_stats()
        summary_file = f"{output_dir}/hit_rate_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nEvaluation complete. Results saved to:")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Summary statistics: {summary_file}")
        
        # Print comparison
        print("\nFeedback Comparison:")
        feedback_comp = summary_stats.get("feedback_comparison", {})
        for category, data in feedback_comp.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for hit_key, value in data.get("hit_rates", {}).items():
                print(f"  {hit_key}: {value:.2f}")
                
    except Exception as e:
        print(f"Error in main process: {str(e)}")
    finally:
        # Clean up connections
        await client_manager.close()
        print("Closed all backend connections")


if __name__ == "__main__":
    asyncio.run(main())