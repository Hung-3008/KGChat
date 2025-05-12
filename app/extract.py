from docling.document_converter import DocumentConverter
from backend.llm.gemini_client import GeminiClient
from pydantic import BaseModel, Field
from pathlib import Path
import os
import asyncio
import json
import re
import glob
from dotenv import load_dotenv
import gc 
from tqdm import tqdm  

load_dotenv()

api_keys = {
    "key1": os.getenv('GEMINI_API_KEY_1'),
    "key2": os.getenv('GEMINI_API_KEY_2'),
    "key3": os.getenv('GEMINI_API_KEY_3'),
    "key4": os.getenv('GEMINI_API_KEY_4')
}


def read_markdown_file_chunked(markdown_path: str, chunk_size=8192) -> str:
    """Read file in chunks to avoid loading entire file in memory"""
    content = []
    with open(markdown_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            content.append(chunk)
    return ''.join(content)


def extract_header_list(markdown_path: str) -> list[str]:
    """Extract headers more efficiently by processing line by line"""
    head_list = []
    with open(markdown_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("##"):
                heading = line[3:].strip()
                head_list.append(heading)
    return head_list


def to_markdown(pdf_path: str, output_markdown_path: str) -> str:
    """Convert PDF to Markdown with memory cleanup"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    folder = Path(output_markdown_path)
    folder.mkdir(parents=True, exist_ok=True)
    file_name = Path(pdf_path).stem 
    markdown_path = folder / f"{file_name}.md"   
    
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown())
        
        del result
        gc.collect()
        
        return str(markdown_path) 
    except Exception as e:
        raise ValueError(f"Failed to convert PDF to Markdown: {str(e)}")


class HeadersClassification(BaseModel):
    content_headers: list[str] = Field(description="List of valuable headers")
    noise_headers: list[str] = Field(description="List of noise headers")


async def classify_headers(head_list, markdown_path: str):
    """Classify headers with reduced memory usage"""
    active_keys = [key for key in [api_keys["key1"], api_keys["key2"]] if key]
    
    
    with tqdm(total=len(active_keys), desc="Trying API keys", leave=False) as pbar:
        for i, key in enumerate(active_keys):
            try:
                pbar.set_description(f"Trying API key {i + 1}")
                client = GeminiClient(api_key=key)
                prompt = f"""Given a list of headers extracted from academic papers and markdown of paper, please separate them into two categories:
1. Valuable Headers: Essential sections that contain academic content such as Abstract, Introduction, Methods, Results, Discussion, Conclusion, Theoretical Framework, etc.

2. Noise Headers: Non-essential elements that don't contain primary research content, such as References, Bibliography, Acknowledgments, page numbers, dates, journal names, author information, etc.

For each header in the following list, identify whether it's a valuable header or noise header, and explain your reasoning briefly:

{head_list}

Finally, provide two clean lists:
- Content Headers: A list of valuable headers.
- Noise Headers: A list of noise headers.
                """
                
                response = await client.generate(prompt=prompt, temperature=0, format=HeadersClassification)
                
                del prompt
                gc.collect()
                
                pbar.update(len(active_keys))  
                return response[0].content_headers, response[0].noise_headers
                
            except Exception as e:
                pbar.update(1)  
                if i == len(active_keys) - 1:
                    raise Exception("All API keys failed.")
                pbar.set_description(f"API key {i + 1} failed, trying key {i + 2}")


def parse_markdown_headers(markdown_path: str, content_headers, noise_headers):
    """Parse markdown with streaming approach to reduce memory usage"""
    result = {
        "contents": [],
        "noise": []
    }
    
    with open(markdown_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    
    header_pattern = re.compile(r'^(## .+?)$', re.MULTILINE)
    header_matches = list(header_pattern.finditer(markdown_text))
    
    for i, match in enumerate(header_matches):
        header_text = match.group(1)[3:].strip()
        start_pos = match.end()
        
        if i < len(header_matches) - 1:
            end_pos = header_matches[i + 1].start()
        else:
            end_pos = len(markdown_text)
            
        content = markdown_text[start_pos:end_pos].strip()
        
        section = {"header": header_text, "content": content}
        
        if any(content_kw.lower() in header_text.lower() for content_kw in content_headers):
            result["contents"].append(section)
        elif (any(noise_kw.lower() in header_text.lower() for noise_kw in noise_headers) or
              bool(re.match(r'^[^a-zA-Z]*$', header_text)) or
              bool(re.search(r'\d+†', header_text)) or
              bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+', header_text))):
            result["noise"].append(section)
        else:
            word_count = len(header_text.split())
            if (word_count <= 3 and '@' not in header_text and '*' not in header_text):
                result["contents"].append(section)
            elif (bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', header_text)) or
                  '@' in header_text or
                  bool(re.match(r'^\d', header_text)) or
                  'correspondence' in header_text.lower()):
                result["noise"].append(section)
            else:
                result["contents"].append(section)
    
    del markdown_text
    gc.collect()
    
    return result


async def process_single_pdf(pdf_path, output_dir):
    """Process a single PDF file with memory cleanup between steps and progress tracking"""
    file_name = Path(pdf_path).name
    try:
        steps = ["Converting to Markdown", "Extracting headers", "Classifying headers", "Parsing markdown", "Saving results"]
        with tqdm(total=len(steps), desc=f"Processing {file_name}", leave=False) as pbar:
            # Step 1: Convert PDF to Markdown
            pbar.set_description(f"{file_name}: {steps[0]}")
            markdown_path = to_markdown(pdf_path, output_dir)
            gc.collect()  # Clear memory after conversion
            pbar.update(1)
            
            # Step 2: Extract headers from Markdown
            pbar.set_description(f"{file_name}: {steps[1]}")
            head_list = extract_header_list(markdown_path)
            gc.collect()  # Clear memory after extraction
            pbar.update(1)
            
            # Step 3: Classify headers using GeminiClient
            pbar.set_description(f"{file_name}: {steps[2]}")
            content_headers, noise_headers = await classify_headers(head_list, markdown_path)
            gc.collect()  # Clear memory after classification
            pbar.update(1)
            
            # Step 4: Parse Markdown based on classified headers
            pbar.set_description(f"{file_name}: {steps[3]}")
            result = parse_markdown_headers(markdown_path, content_headers, noise_headers)
            pbar.update(1)
            
            # Step 5: Save the result to a JSON file
            pbar.set_description(f"{file_name}: {steps[4]}")
            output_json_path = Path(output_dir) / f"{Path(pdf_path).stem}_result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            pbar.update(1)
        
        del result, head_list, content_headers, noise_headers
        gc.collect()
        
        return True
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return False


async def main():
    input_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/input/001"
    output_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/output/001"
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
            
        print(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        successful_files = 0
        with tqdm(total=len(pdf_files), desc="Processing PDF files") as pbar:
            for pdf_path in pdf_files:
                success = await process_single_pdf(pdf_path, output_dir)
                if success:
                    successful_files += 1
                gc.collect()
                pbar.update(1)
            
        print(f"\nProcessed {successful_files} out of {len(pdf_files)} PDF files successfully.")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())