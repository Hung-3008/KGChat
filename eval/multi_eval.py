import requests
import time
from datasets import Dataset
import pandas as pd
import os
import sys
import logging
from pydantic import BaseModel, Field
import asyncio
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(".")
from app.backend.llm.gemini_client import GeminiClient

API_URL = "http://localhost:8000/api/query"

class EvaluationResult(BaseModel):
    is_correct: int = Field()

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
        self.logger = logging.getLogger(__name__)
    
    def get_current_key(self) -> str:
        key_name = self.key_pattern.format(self.current_key_index)
        api_key = os.getenv(key_name)
        
        if not api_key:
            self.logger.warning(f"API key {key_name} not found in environment variables")
            return None
            
        return api_key
    
    def rotate_key(self) -> str:
        for _ in range(self.max_keys):
            self.current_key_index = (self.current_key_index % self.max_keys) + 1
            
            key_name = self.key_pattern.format(self.current_key_index)
            api_key = os.getenv(key_name)
            
            if api_key:
                self.logger.info(f"Rotated to API key {key_name}")
                return api_key
        
        self.logger.error("No valid API keys found after trying all options")
        return None

async def evaluate_answer(gemini_client, key_manager, question, correct_answer, model_answer):
    prompt = f"""
    You are an evaluation system for multiple-choice answers. 
    
    Question:
    {question}
    
    Correct answer: {correct_answer}
    
    Model's answer: {model_answer}
    
    Your task is to determine if the model's answer correctly identifies the same answer choice as the correct answer.
    
    Rules:
    - The model might not format its answer exactly like the correct answer
    - The model might give additional explanations
    - Focus on whether the model correctly identifies the same option (A, B, C, or D)
    - Return 1 if the model's answer correctly identifies the same option as the correct answer
    - Return 0 if the model's answer does not correctly identify the same option
    
    Do not provide explanations, just evaluate and return the binary result.
    """
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = await gemini_client.generate(
                prompt=prompt,
                format=EvaluationResult
            )
            
            if isinstance(response, list) and len(response) > 0:
                result = response[0].is_correct
            else:
                result = 0
                
            return result
            
        except Exception as e:
            logger.error(f"Error during evaluation (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                new_key = key_manager.rotate_key()
                if new_key:
                    gemini_client.api_key = new_key
                    os.environ["GEMINI_API_KEY"] = new_key
                    logger.info(f"Switched to API key index {key_manager.current_key_index} for next attempt")
                else:
                    logger.error("No more API keys available to try")
                    break
                
                await asyncio.sleep(2)
    
    logger.warning("All evaluation attempts failed")
    return 0

def ask_question(question, response_type="concise"):
    try:
        guided_question = f"""
        This is a multiple-choice question about diabetes. Please read the question carefully and select the most appropriate answer option (A, B, C, or D).

        IMPORTANT INSTRUCTIONS:
        - Provide ONLY the letter and text of your selected answer (e.g., "B) Family history of diabetes")
        - Do NOT include any explanations, reasoning, or additional text
        - Do NOT restate the question
        - Just directly state which option is correct

        QUESTION:
        {question}
        """
        
        payload = {
            "query": guided_question,
            "conversation_history": [],
            "response_type": response_type
        }
        
        response = requests.post(API_URL, json=payload, timeout=60)

        time.sleep(4)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
        else:
            logger.warning(f"Error: Status code {response.status_code}")
            return f"Error: Status code {response.status_code}"
            
    except requests.exceptions.Timeout:
        logger.warning("Error: Request timed out")
        return "Error: Request timed out"
    except requests.exceptions.ConnectionError:
        logger.warning("Error: Could not connect to the server")
        return "Error: Could not connect to the server"
    except Exception as e:
        logger.error(f"Error querying API: {str(e)}")
        return f"Error: {str(e)}"

def create_excel_with_formatting(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Evaluation Results')
    
    workbook = writer.book
    worksheet = writer.sheets['Evaluation Results']
    
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    correct_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    incorrect_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    worksheet.column_dimensions['A'].width = 15
    worksheet.column_dimensions['B'].width = 25
    worksheet.column_dimensions['C'].width = 25
    worksheet.column_dimensions['D'].width = 15
    
    for col_num, column_title in enumerate(df.columns, 1):
        cell = worksheet.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        
    for row_num, value in enumerate(df['is_correct'], 2):
        cell = worksheet.cell(row=row_num, column=4)
        if value == 1:
            cell.fill = correct_fill
            cell.value = "1"
        else:
            cell.fill = incorrect_fill
            cell.value = "0"
    
    writer.close()
    
    logger.info(f"Excel file created at {output_path} with formatting")

async def process_dataset(dataset, gemini_client, key_manager, num_samples=None, batch_size=100, save_prefix='diabetes_mc_results'):
    results = []
    
    total_examples = len(dataset['input'])
    examples_to_process = min(total_examples, num_samples) if num_samples else total_examples
    
    logger.info(f"Starting processing of {examples_to_process} questions")
    
    # Create output directories
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H_%M_%S")
    
    batch_dir = f"eval/batches/{date_str}/{time_str}"
    os.makedirs(batch_dir, exist_ok=True)
    
    for i in range(examples_to_process):
        question = dataset['input'][i]
        correct_answer = dataset['output'][i]
        
        logger.info(f"Processing question {i+1}/{examples_to_process}")
        
        model_answer = ask_question(question)
        logger.debug(f"Model answered: {model_answer}")
        
        is_correct = await evaluate_answer(
            gemini_client, 
            key_manager, 
            question, 
            correct_answer, 
            model_answer
        )
        
        results.append({
            'question': question,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'is_correct': is_correct
        })
        
        if (i + 1) % batch_size == 0 or i == examples_to_process - 1:
            batch_number = (i + 1) // batch_size if (i + 1) % batch_size == 0 else ((i + 1) // batch_size) + 1
            batch_df = pd.DataFrame(results)
            batch_filename = f"{batch_dir}/{save_prefix}_batch_{batch_number}.xlsx"
            
            create_excel_with_formatting(batch_df, batch_filename)
            logger.info(f"Saved batch {batch_number} with {len(batch_df)} questions to {batch_filename}")
        
        time.sleep(4)
    
    logger.info(f"Completed processing all {examples_to_process} questions")
    return pd.DataFrame(results)

async def main():
    logger.info("Starting evaluation process")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    key_manager = GeminiKeyManager(current_key_index=4, max_keys=6)
    api_key = key_manager.get_current_key()
    
    if not api_key:
        logger.error("No valid Gemini API key found in environment variables")
        return
    
    gemini_client = GeminiClient(api_key=api_key, model_name="gemini-2.0-flash")
    logger.info(f"Initialized Gemini client with API key index {key_manager.current_key_index}")
    
    arrow_file_path = "/home/hung/Documents/hung/code/KG_Hung/KGChat/eval/multiple_choice/diabetes_instruct_v8-train.arrow"
    logger.info(f"Loading dataset from {arrow_file_path}")
    
    try:
        dataset = Dataset.from_file(arrow_file_path)
        dataset_dict = dataset.to_dict()
        logger.info(f"Dataset loaded successfully with {len(dataset_dict['input'])} questions")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    # Create output directories with date and time
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H_%M_%S")
    
    output_dir = f"eval/multiple_choice/{date_str}/{time_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = 1500 
    logger.info(f"Will process {num_samples} samples")
    
    results_df = await process_dataset(
        dataset=dataset_dict, 
        gemini_client=gemini_client,
        key_manager=key_manager,
        num_samples=num_samples,
        batch_size=100,  
        save_prefix='diabetes_mc_results'
    )
    
    # Create final Excel file with formatting
    final_output_path = f"{output_dir}/diabetes_mc_results_complete.xlsx"
    create_excel_with_formatting(results_df, final_output_path)
    logger.info(f"Saved complete results to {final_output_path}")
    
    # Calculate and log summary statistics
    total_questions = len(results_df)
    correct_count = results_df['is_correct'].sum()
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    logger.info("\nResults Summary:")
    logger.info(f"Total questions processed: {total_questions}")
    logger.info(f"Correct answers: {correct_count} ({accuracy:.2f}%)")
    
    # Create summary sheet
    summary_df = pd.DataFrame([{
        'Total Questions': total_questions,
        'Correct Answers': correct_count,
        'Accuracy': f"{accuracy:.2f}%",
        'Date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    summary_file = f"{output_dir}/diabetes_mc_results_summary.xlsx"
    summary_df.to_excel(summary_file, index=False)
    logger.info(f"Saved summary statistics to {summary_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)