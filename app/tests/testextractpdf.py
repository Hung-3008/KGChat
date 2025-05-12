from app.backend.llm.gemini_client import GeminiClient
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY_1")
client = GeminiClient(
    api_key= API_KEY
    model_name= "gemini-2.0-flash"
    max_retries= 3
)
pdf_path = ""
try:
    extracted_text = client.extract_pdf_raw_text(pdf_path)
    print("Extracted text:")
    print(extracted_text)
except Exception as e:
    print(f"Error: {str(e)}")