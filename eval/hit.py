import json
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
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
    """Structured response for answer evaluation"""
    result: str = Field(description="One of: 'accurate', 'incorrect', 'missing'")
    reasoning: str = Field(description="Explanation of the evaluation decision")


class CommentorResponse(BaseModel):
    """Structured response for feedback generation"""
    feedback: str = Field(description="Constructive feedback to improve the answer")


class GeminiKeyManager:
    """Manages rotation of multiple Gemini API keys to handle rate limits"""
    
    def __init__(self, current_key_index: int = 1, max_keys: int = 6, key_pattern: str = "GEMINI_API_KEY_{}"):
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
    """Manages all backend client connections"""
    
    def __init__(self, gemini_key_manager):
        self.gemini_key_manager = gemini_key_manager
        self.clients = None
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return self.clients
            
        load_dotenv()
        
        # Initialize all backend clients
        neo4j_client = Neo4jClient(
            uri=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        vector_db_client = VectorDBClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"), 
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        
        ollama_client = OllamaClient(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        )
        
        gemini_client = GeminiClient(
            api_key=self.gemini_key_manager.get_current_key(), 
            model_name="gemini-2.0-flash"
        )
        
        # Verify connectivity
        if not await neo4j_client.verify_connectivity():
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
        new_key = self.gemini_key_manager.rotate_key()
        if new_key:
            self.clients["gemini_client"] = GeminiClient(api_key=new_key, model_name="gemini-2.0-flash")
            return self.clients["gemini_client"]
        return None
    
    async def close(self):
        if self.clients and "neo4j_client" in self.clients:
            await self.clients["neo4j_client"].close()
        self.initialized = False


class BackendAPIClient:
    """Adapter that uses backend services directly instead of HTTP API calls"""
    
    def __init__(self, client_manager: BackendClientManager, max_retries: int = 3):
        self.client_manager = client_manager
        self.max_retries = max_retries
    
    async def get_answer(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Get answer using backend services with retry logic and key rotation"""
        if conversation_history is None:
            conversation_history = []
        
        clients = await self.client_manager.initialize()
        
        for attempt in range(self.max_retries):
            try:
                result = await run_query(
                    query=prompt,
                    conversation_history=conversation_history,
                    clients=clients,
                    grounding=False,
                    language="English"
                )
                
                if result and "response" in result:
                    return result["response"]
                else:
                    print(f"No response received on attempt {attempt + 1}")
                    
            except Exception as e:
                error_str = str(e).lower()
                print(f"Error on attempt {attempt + 1}/{self.max_retries}: {error_str}")
                
                # Handle rate limit errors by rotating API keys
                if any(keyword in error_str for keyword in ["resource_exhausted", "rate limit", "429", "quota"]):
                    new_client = self.client_manager.rotate_gemini_key()
                    if new_client:
                        clients["gemini_client"] = new_client
                        await asyncio.sleep(2)
                    else:
                        print("No more valid Gemini API keys available")
                        break
                else:
                    await asyncio.sleep(1)
                    
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2)
        
        return "Error: Failed to get response from backend services"


class HitRateCalculator:
    """
    Calculates Hit@K metrics for evaluating retrieval-augmented generation systems.
    
    Hit Rate Logic Explanation:
    
    Hit@K measures the percentage of questions where the correct/satisfactory answer 
    is found within the first K attempts. This is particularly useful for:
    
    1. Evaluating RAG (Retrieval-Augmented Generation) systems
    2. Measuring the effectiveness of iterative refinement
    3. Assessing feedback mechanisms
    
    Calculation:
    - For each question, we make up to max_attempts tries
    - Each attempt is evaluated as "hit" (correct) or "miss" (incorrect)
    - Hit@1 = percentage of questions answered correctly on first attempt
    - Hit@2 = percentage of questions answered correctly within first 2 attempts
    - Hit@K = percentage of questions answered correctly within first K attempts
    
    Example:
    - 100 questions total
    - 60 answered correctly on attempt 1 → Hit@1 = 0.60
    - 20 more answered correctly on attempt 2 → Hit@2 = 0.80
    - 10 more answered correctly on attempt 3 → Hit@3 = 0.90
    
    The cumulative nature means Hit@K >= Hit@(K-1) always.
    """
    
    def __init__(self, api_client, evaluator_client, max_attempts: int = 3):
        self.api_client = api_client
        self.evaluator_client = evaluator_client  # Separate Gemini client for evaluation
        self.max_attempts = max_attempts
        self.results = {}
    
    async def calculate_hit_rates(self, dataset: List[Dict[str, Any]], use_feedback: bool = False) -> Dict[str, Any]:
        """
        Calculate hit rates for a dataset with detailed logging of the process
        
        Args:
            dataset: List of questions with ground truth answers
            use_feedback: Whether to use feedback for iterative improvement
            
        Returns:
            Dictionary containing hit rates and detailed results
        """
        print(f"\n{'='*60}")
        print(f"STARTING HIT RATE CALCULATION")
        print(f"{'='*60}")
        print(f"Dataset size: {len(dataset)} questions")
        print(f"Max attempts per question: {self.max_attempts}")
        print(f"Feedback enabled: {use_feedback}")
        print(f"{'='*60}\n")
        
        question_results = []
        
        # Track hits by attempt number (cumulative)
        hits_by_attempt = [0] * self.max_attempts
        
        for i, question_data in enumerate(dataset):
            print(f"\n--- Processing Question {i+1}/{len(dataset)} ---")
            print(f"Question: {question_data['question'][:100]}...")
            
            result = await self._process_single_question(
                question_data=question_data,
                question_id=f"q{i+1}",
                use_feedback=use_feedback
            )
            
            question_results.append(result)
            
            # Update cumulative hit counters
            if result["hit_at"] > 0:
                # If question was answered correctly at attempt N,
                # it counts as a hit for Hit@N, Hit@(N+1), Hit@(N+2), etc.
                for attempt_idx in range(result["hit_at"] - 1, self.max_attempts):
                    hits_by_attempt[attempt_idx] += 1
            
            print(f"Result: {'HIT' if result['hit_at'] > 0 else 'MISS'} at attempt {result['hit_at'] if result['hit_at'] > 0 else 'N/A'}")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Calculate hit rates
        total_questions = len(dataset)
        hit_rates = {}
        
        print(f"\n{'='*60}")
        print(f"HIT RATE CALCULATION RESULTS")
        print(f"{'='*60}")
        
        for i in range(self.max_attempts):
            hit_rate = hits_by_attempt[i] / total_questions
            hit_rates[f"hit_rate@{i+1}"] = hit_rate
            
            print(f"Hit@{i+1}: {hits_by_attempt[i]}/{total_questions} = {hit_rate:.3f} ({hit_rate*100:.1f}%)")
        
        print(f"{'='*60}")
        
        # Verify cumulative property
        print(f"\nVerifying cumulative property:")
        for i in range(1, self.max_attempts):
            current_rate = hit_rates[f"hit_rate@{i+1}"]
            previous_rate = hit_rates[f"hit_rate@{i}"]
            is_valid = current_rate >= previous_rate
            print(f"Hit@{i+1} >= Hit@{i}: {current_rate:.3f} >= {previous_rate:.3f} = {is_valid}")
            
            if not is_valid:
                print(f"WARNING: Cumulative property violated!")
        
        results_summary = {
            "total_questions": total_questions,
            "hit_rates": hit_rates,
            "hits_by_attempt": hits_by_attempt,
            "use_feedback": use_feedback,
            "question_results": question_results,
            "evaluation_timestamp": time.time()
        }
        
        # Store results with timestamp
        result_key = f"feedback_{use_feedback}_{int(time.time())}"
        self.results[result_key] = results_summary
        
        return results_summary
    
    async def _process_single_question(self, question_data: Dict[str, Any], 
                                     question_id: str, use_feedback: bool = False) -> Dict[str, Any]:
        """Process a single question through multiple attempts until hit or max attempts reached"""
        
        question = question_data["question"]
        ground_truth = question_data["ground_truth"]
        
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "attempts": [],
            "hit_at": 0  # 0 means no hit, 1-N means hit at attempt N
        }
        
        conversation_history = []
        
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"  Attempt {attempt_num}/{self.max_attempts}")
            
            # Skip if already got a hit
            if result["hit_at"] > 0:
                result["attempts"].append({
                    "attempt": attempt_num,
                    "skipped": True,
                    "reason": f"Already got hit at attempt {result['hit_at']}"
                })
                continue
            
            # Prepare prompt for this attempt
            if attempt_num == 1:
                prompt = question
            else:
                # Use feedback if available and enabled
                if use_feedback and result["attempts"]:
                    prev_attempt = result["attempts"][-1]
                    if "feedback" in prev_attempt:
                        prompt = f"{question}\n\nPrevious answer: {prev_attempt['response']}\nFeedback: {prev_attempt['feedback']}\nPlease provide an improved answer."
                    else:
                        prompt = f"{question}\n\nPlease provide a more accurate answer."
                else:
                    prompt = f"{question}\n\nPlease provide a more detailed answer."
            
            # Get response from backend
            response = await self.api_client.get_answer(
                prompt=prompt,
                conversation_history=conversation_history
            )
            
            # Evaluate the response
            is_hit, eval_result, reasoning = await self._evaluate_response(
                question=question,
                ground_truth=ground_truth,
                response=response
            )
            
            attempt_result = {
                "attempt": attempt_num,
                "prompt": prompt,
                "response": response,
                "is_hit": is_hit,
                "evaluation": eval_result,
                "reasoning": reasoning
            }
            
            print(f"    Response length: {len(response)} chars")
            print(f"    Evaluation: {eval_result}")
            print(f"    Is Hit: {is_hit}")
            
            # Record hit
            if is_hit and result["hit_at"] == 0:
                result["hit_at"] = attempt_num
                print(f"    *** HIT ACHIEVED at attempt {attempt_num} ***")
            
            # Generate feedback for next attempt if needed
            if not is_hit and attempt_num < self.max_attempts and use_feedback:
                feedback = await self._generate_feedback(
                    question=question,
                    ground_truth=ground_truth,
                    response=response,
                    reasoning=reasoning
                )
                attempt_result["feedback"] = feedback
                print(f"    Generated feedback: {feedback[:100]}...")
            
            result["attempts"].append(attempt_result)
            
            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
        
        return result
    
    async def _evaluate_response(self, question: str, ground_truth: Dict[str, Any], 
                               response: str) -> Tuple[bool, str, str]:
        """Evaluate if a response is correct/satisfactory"""
        
        prompt = f"""
### Question: {question}
### Ground Truth: {self._format_ground_truth(ground_truth)}
### Model Response: {response}

### Task: 
Evaluate whether the model response correctly answers the question based on the ground truth.
- Return "accurate" if the response is correct and complete
- Return "incorrect" if the response is wrong 
- Return "missing" if the response is partially correct but missing key information

Provide your reasoning for the evaluation.
        """.strip()
        
        try:
            evaluation = await self.evaluator_client.generate(
                prompt=prompt,
                format=EvaluatorResponse
            )
            
            # Handle different response formats
            if isinstance(evaluation, list) and len(evaluation) > 0:
                eval_result = evaluation[0]
            else:
                eval_result = evaluation
            
            is_hit = eval_result.result.lower() == "accurate"
            return is_hit, eval_result.result, eval_result.reasoning
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            # Fallback to simple string matching
            return self._simple_evaluation(response, ground_truth)
    
    def _simple_evaluation(self, response: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Fallback evaluation using simple string matching"""
        
        if "answer" in ground_truth:
            expected = ground_truth["answer"].lower().strip()
            is_hit = expected in response.lower()
            return is_hit, "accurate" if is_hit else "incorrect", f"Simple string match for '{expected}'"
        
        elif "required_elements" in ground_truth:
            elements = ground_truth["required_elements"]
            required_count = ground_truth.get("required_count", len(elements))
            
            found_elements = [elem for elem in elements if elem.lower() in response.lower()]
            is_hit = len(found_elements) >= required_count
            
            return is_hit, "accurate" if is_hit else "missing", f"Found {len(found_elements)}/{required_count} required elements"
        
        return False, "incorrect", "Unable to evaluate"
    
    def _format_ground_truth(self, ground_truth: Dict[str, Any]) -> str:
        """Format ground truth for display in evaluation prompt"""
        if "answer" in ground_truth:
            return ground_truth["answer"]
        elif "required_elements" in ground_truth:
            elements = ground_truth["required_elements"]
            count = ground_truth.get("required_count", len(elements))
            return f"Must include at least {count} of: {', '.join(elements)}"
        return str(ground_truth)
    
    async def _generate_feedback(self, question: str, ground_truth: Dict[str, Any], 
                               response: str, reasoning: str) -> str:
        """Generate constructive feedback for improvement"""
        
        prompt = f"""
Question: {question}
Expected Answer: {self._format_ground_truth(ground_truth)}
Current Response: {response}
Evaluation: {reasoning}

Provide constructive feedback to help improve the response. Be specific about what's missing or incorrect, but don't give away the complete answer. Keep feedback concise (2-3 sentences).
        """.strip()
        
        try:
            feedback_response = await self.evaluator_client.generate(
                prompt=prompt,
                format=CommentorResponse
            )
            
            if isinstance(feedback_response, list) and len(feedback_response) > 0:
                return feedback_response[0].feedback
            else:
                return feedback_response.feedback
                
        except Exception as e:
            print(f"Error generating feedback: {str(e)}")
            return f"Your answer needs improvement. {reasoning} Please be more specific and complete."
    
    def save_results(self, filepath: str) -> None:
        """Save all evaluation results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filepath}")
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get a summary comparing feedback vs no-feedback performance"""
        
        if len(self.results) < 2:
            return {"error": "Need at least 2 evaluation runs to compare"}
        
        with_feedback = None
        without_feedback = None
        
        for key, result in self.results.items():
            if result.get("use_feedback", False):
                with_feedback = result
            else:
                without_feedback = result
        
        if not with_feedback or not without_feedback:
            return {"error": "Need both feedback and no-feedback results to compare"}
        
        comparison = {
            "without_feedback": without_feedback["hit_rates"],
            "with_feedback": with_feedback["hit_rates"],
            "improvement": {}
        }
        
        for key in without_feedback["hit_rates"]:
            improvement = with_feedback["hit_rates"][key] - without_feedback["hit_rates"][key]
            comparison["improvement"][key] = improvement
        
        return comparison


def load_dataset_from_arrow(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from Arrow file and convert to hit rate evaluation format
    
    Args:
        data_path: Path to the Arrow dataset file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of formatted question-answer pairs for hit rate evaluation
    """
    try:
        from datasets import Dataset
        
        print(f"Loading dataset from: {data_path}")
        dataset = Dataset.from_file(data_path)
        data_dict = dataset.to_dict()
        
        print(f"Dataset keys: {list(data_dict.keys())}")
        
        # Check available keys and determine question/answer fields
        available_keys = set(data_dict.keys())
        
        # Common field name mappings
        question_field = None
        answer_field = None
        
        # Try to identify question field
        for possible_q in ['question', 'input', 'query', 'prompt', 'text']:
            if possible_q in available_keys:
                question_field = possible_q
                break
        
        # Try to identify answer field  
        for possible_a in ['answer', 'output', 'response', 'target', 'label']:
            if possible_a in available_keys:
                answer_field = possible_a
                break
        
        if not question_field or not answer_field:
            raise ValueError(f"Could not identify question/answer fields. Available keys: {available_keys}")
        
        print(f"Using '{question_field}' as question field and '{answer_field}' as answer field")
        
        # Get the data
        questions = data_dict[question_field]
        answers = data_dict[answer_field]
        
        # Limit samples if specified
        if max_samples and max_samples < len(questions):
            questions = questions[:max_samples]
            answers = answers[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"Processing {len(questions)} question-answer pairs")
        
        # Convert to hit rate evaluation format
        formatted_dataset = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Format ground truth - using the provided answer as exact match
            ground_truth = {
                "answer": str(answer).strip()
            }
            
            # Alternative: If answers are long, we can break them into key elements
            # This is useful for open-ended questions where partial matches should count
            if len(str(answer)) > 100:  # For longer answers, extract key elements
                # Split by sentences/points and use as required elements
                import re
                sentences = re.split(r'[.!?]+', str(answer))
                elements = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(elements) > 1:
                    ground_truth = {
                        "required_elements": elements,
                        "required_count": max(1, len(elements) // 2)  # Require at least half
                    }
            
            formatted_dataset.append({
                "id": f"q{i+1}",
                "question": str(question).strip(),
                "ground_truth": ground_truth
            })
        
        print(f"Successfully formatted {len(formatted_dataset)} samples")
        
        # Show sample for verification
        if formatted_dataset:
            sample = formatted_dataset[0]
            print(f"\nSample formatted entry:")
            print(f"  ID: {sample['id']}")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Ground Truth Type: {'exact_answer' if 'answer' in sample['ground_truth'] else 'required_elements'}")
            if 'answer' in sample['ground_truth']:
                print(f"  Expected Answer: {sample['ground_truth']['answer'][:100]}...")
            else:
                print(f"  Required Elements: {len(sample['ground_truth']['required_elements'])} elements")
        
        return formatted_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def create_test_dataset() -> List[Dict[str, Any]]:
    """Create a small test dataset for quick testing (fallback)"""
    
    return [
        {
            "question": "What are the main symptoms of type 2 diabetes?",
            "ground_truth": {
                "required_elements": [
                    "increased thirst", 
                    "frequent urination", 
                    "fatigue", 
                    "blurred vision", 
                    "slow healing wounds"
                ],
                "required_count": 3
            }
        },
        {
            "question": "How is diabetes diagnosed?",
            "ground_truth": {
                "required_elements": [
                    "fasting glucose test",
                    "A1C test", 
                    "glucose tolerance test",
                    "random glucose test"
                ],
                "required_count": 2
            }
        }
    ]


async def main():
    """Main execution function with comprehensive hit rate analysis using external dataset"""
    
    print("Hit Rate Calculator - Backend Integration")
    print("=" * 50)
    
    load_dotenv()
    
    data_path = "/home/hung/Documents/hung/code/KG_Hung/KGChat/eval/hitrate/diabetes_qa_dataset-train.arrow"
    max_samples = 5  
    key_manager = GeminiKeyManager(current_key_index=1, max_keys=6)
    client_manager = BackendClientManager(key_manager)
    
    try:
        print("Initializing backend clients...")
        api_client = BackendAPIClient(client_manager, max_retries=3)
        
        evaluator_client = GeminiClient(
            api_key=key_manager.get_current_key(),
            model_name="gemini-2.0-flash"
        )
        
        calculator = HitRateCalculator(
            api_client=api_client,
            evaluator_client=evaluator_client,
            max_attempts=3
        )
        
        print(f"\nLoading dataset from: {data_path}")
        dataset = load_dataset_from_arrow(data_path, max_samples=max_samples)
        
        if not dataset:
            print("Failed to load dataset, falling back to test dataset...")
            dataset = create_test_dataset()
        
        print(f"Dataset loaded successfully: {len(dataset)} questions")
        
        ground_truth_types = {}
        for item in dataset:
            gt_type = "exact_answer" if "answer" in item["ground_truth"] else "required_elements"
            ground_truth_types[gt_type] = ground_truth_types.get(gt_type, 0) + 1
        
        print(f"Ground truth types: {ground_truth_types}")
        
        output_dir = f"eval/hit_rate_analysis/{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        dataset_info = {
            "source_path": data_path,
            "total_samples": len(dataset),
            "max_samples_used": max_samples,
            "ground_truth_distribution": ground_truth_types,
            "sample_questions": [item["question"] for item in dataset[:3]]
        }
        
        with open(f"{output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        evaluation_results = {}
        
        for use_feedback in [False, True]:
            mode = "WITH FEEDBACK" if use_feedback else "WITHOUT FEEDBACK"
            print(f"\n{'='*20} {mode} {'='*20}")
            
            try:
                results = await calculator.calculate_hit_rates(
                    dataset=dataset,
                    use_feedback=use_feedback
                )
                
                evaluation_results[f"feedback_{use_feedback}"] = results
                
                intermediate_file = f"{output_dir}/results_{mode.lower().replace(' ', '_')}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                print(f"Error in {mode} evaluation: {str(e)}")
                continue
        
        results_file = f"{output_dir}/hit_rate_detailed_results.json"
        calculator.save_results(results_file)
        
        if len(evaluation_results) >= 2:
            print("\nGenerating comparison analysis...")
            comparison = calculator.get_comparison_summary()
            comparison_file = f"{output_dir}/hit_rate_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print(f"\n{'='*60}")
            print("FINAL COMPARISON: FEEDBACK vs NO FEEDBACK")
            print(f"{'='*60}")
            print(f"Dataset: {len(dataset)} questions from {os.path.basename(data_path)}")
            print(f"Max attempts per question: 3")
            print(f"{'='*60}")
            
            if "error" not in comparison:
                for metric in ["hit_rate@1", "hit_rate@2", "hit_rate@3"]:
                    without = comparison["without_feedback"][metric]
                    with_fb = comparison["with_feedback"][metric]
                    improvement = comparison["improvement"][metric]
                    
                    print(f"{metric}:")
                    print(f"  Without feedback: {without:.3f} ({without*100:.1f}%)")
                    print(f"  With feedback:    {with_fb:.3f} ({with_fb*100:.1f}%)")
                    print(f"  Improvement:      {improvement:+.3f} ({improvement*100:+.1f}%)")
                    
                    if improvement > 0.1:
                        print(f"  Significant improvement with feedback")
                    elif improvement > 0.05:
                        print(f"  Moderate improvement with feedback")
                    elif improvement > 0:
                        print(f"  Small improvement with feedback")
                    elif improvement == 0:
                        print(f"  No change with feedback")
                    else:
                        print(f"  Feedback decreased performance")
                    print()
                
                avg_improvement = sum(comparison["improvement"].values()) / len(comparison["improvement"])
                print(f"Average improvement across all metrics: {avg_improvement:+.3f}")
                
                if avg_improvement > 0.05:
                    print("CONCLUSION: Feedback mechanism shows significant benefit")
                elif avg_improvement > 0:
                    print("CONCLUSION: Feedback mechanism shows positive benefit")
                else:
                    print("CONCLUSION: Feedback mechanism shows no clear benefit")
            
            else:
                print("Could not generate comparison - missing evaluation results")
        
        else:
            print("Could not generate comparison - need both feedback and no-feedback results")
        
        print(f"\n{'='*60}")
        print("RESULTS SAVED TO:")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset info: dataset_info.json")
        print(f"Detailed results: hit_rate_detailed_results.json") 
        if len(evaluation_results) >= 2:
            print(f"Comparison analysis: hit_rate_comparison.json")
        print(f"{'='*60}")
        
        total_questions = len(dataset)
        total_attempts = sum(len(evaluation_results) * total_questions * 3 for _ in evaluation_results)  # approximate
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"   Questions processed: {total_questions}")
        print(f"   Total API calls: ~{total_attempts}")
        print(f"   Evaluation modes: {len(evaluation_results)}")
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up connections...")
        await client_manager.close()
        print("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())