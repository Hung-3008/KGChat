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
    """Convert PDF to Markdown"""
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
        
        return str(markdown_path) 
    except Exception as e:
        raise ValueError(f"Failed to convert PDF to Markdown: {str(e)}")


class HeadersClassification(BaseModel):
    content_headers: list[str] = Field(description="List of valuable headers")
    noise_headers: list[str] = Field(description="List of noise headers")


async def classify_headers(head_list, markdown_path: str):
    """Classify headers"""
    active_keys = [key for key in [api_keys["key1"], api_keys["key2"]] if key]
    
    for i, key in enumerate(active_keys):
        try:
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
            
            return response[0].content_headers, response[0].noise_headers
            
        except Exception as e:
            if i == len(active_keys) - 1:
                raise Exception("All API keys failed.")


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
    
    return result


async def process_single_pdf(pdf_path, output_dir):
    """Process a single PDF file"""
    file_name = Path(pdf_path).name
    try:
        # Step 1: Convert PDF to Markdown
        print(f"Converting {file_name} to Markdown")
        markdown_path = to_markdown(pdf_path, output_dir)
        
        # Step 2: Extract headers from Markdown
        print(f"Extracting headers from {file_name}")
        head_list = extract_header_list(markdown_path)
        
        # Step 3: Classify headers using GeminiClient
        print(f"Classifying headers for {file_name}")
        content_headers, noise_headers = await classify_headers(head_list, markdown_path)
        
        # Step 4: Parse Markdown based on classified headers
        print(f"Parsing markdown for {file_name}")
        result = parse_markdown_headers(markdown_path, content_headers, noise_headers)
        
        # Step 5: Save the result to a JSON file
        print(f"Saving results for {file_name}")
        output_json_path = Path(output_dir) / f"{Path(pdf_path).stem}_result.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        return True
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return False


async def main():
    input_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/input/002"
    output_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/output/002"
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
            
        print(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        md_files = glob.glob(os.path.join(output_dir, "*.md"))
        processed_files = set()
        
        for md_path in md_files:
            md_basename = os.path.basename(md_path)
            md_filename_without_ext = os.path.splitext(md_basename)[0]
            processed_files.add(md_filename_without_ext)
        
        unprocessed_pdf_files = []
        for pdf_path in pdf_files:
            pdf_basename = os.path.basename(pdf_path)
            pdf_filename_without_ext = os.path.splitext(pdf_basename)[0]
            
            if pdf_filename_without_ext not in processed_files:
                unprocessed_pdf_files.append(pdf_path)
        
        print(f"Found {len(unprocessed_pdf_files)} unprocessed PDF files out of {len(pdf_files)} total")
        
        successful_files = 0
        for pdf_path in unprocessed_pdf_files:
            print(f"Processing {os.path.basename(pdf_path)}")
            success = await process_single_pdf(pdf_path, output_dir)
            if success:
                successful_files += 1
            
        print(f"\nProcessed {successful_files} out of {len(unprocessed_pdf_files)} PDF files successfully.")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())