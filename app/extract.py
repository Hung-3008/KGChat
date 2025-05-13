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
from tqdm import tqdm

# Load environment variables
load_dotenv()
api_key1 = os.getenv('GEMINI_API_KEY_1')
api_key2 = os.getenv('GEMINI_API_KEY_2')
api_key3 = os.getenv('GEMINI_API_KEY_3')
api_key4 = os.getenv('GEMINI_API_KEY_4')


def read_markdown_file(markdown_path: str) -> str:
    """Read content from a markdown file."""
    with open(markdown_path, 'r', encoding='utf-8') as file:
        return file.read()


def to_markdown(pdf_path: str, output_markdown_path: str) -> str:
    """Convert PDF to markdown format."""
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


def take_head_list(markdown_path: str) -> list[str]:
    """Extract headers from markdown file."""
    head_list: list[str] = []
    with open(markdown_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("##"):
                heading = line[3:].strip()
                head_list.append(heading)
    return head_list


class HeadersClassification(BaseModel):
    """Model for classifying headers."""
    content_headers: list[str] = Field(description="List of valuable headers")
    noise_headers: list[str] = Field(description="List of noise headers")


async def clasify_headers(head_list, markdown_path: str):
    """Classify headers using Gemini API."""
    api_keys = [
        api_key1,
        api_key2,
        api_key3,
        api_key4
    ]
    
    for i, key in enumerate(api_keys):
        try:
            print(f"Using API key {i + 1}: {key}")
            client = GeminiClient(api_key=key)
            prompt = f"""Given a list of headers extracted from academic papers and markdown of paper, please separate them into two categories:

            1. Valuable Headers: Essential sections that contain academic content such as Abstract, Introduction, Methods, Results, Discussion, Conclusion, Theoretical Framework, etc.

            2. Noise Headers: Non-essential elements that don't contain primary research content, such as References, Bibliography, Acknowledgments, page numbers, dates, journal names, author information, etc.

            For each header in the following list, identify whether it's a valuable header or noise header, and explain your reasoning briefly:

            <<Header List>>
            {head_list}
            <<End of Header List>>

            Finally, provide two clean lists:
            - Content Headers: A list of valuable headers.
            - Noise Headers: A list of noise headers.
            """
            response = await client.generate(prompt=prompt, temperature=0, format=HeadersClassification)
            content_headers = response[0].content_headers
            noise_headers = response[0].noise_headers
            return content_headers, noise_headers
        except Exception as e:
            print(f"Error with API key {i + 1}: {str(e)}")
            if i == len(api_keys) - 1:
                raise Exception("All API keys have been tried and failed.")
            print(f"Retrying with next API key ({i + 2})...")


def parse_markdown_headers(markdown_path: str, content_headers, noise_headers):
    """Parse markdown headers and categorize content."""
    markdown_text = read_markdown_file(markdown_path)
    result = {
        "contents": [],
        "noise": []
    }
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
        section = {
            "header": header_text,
            "content": content
        }
        
        categorized = False
        
        # Check if header is in content headers
        for content_keyword in content_headers:
            if content_keyword.lower() in header_text.lower():
                result["contents"].append(section)
                categorized = True
                break
        
        # Check if header is in noise headers
        if not categorized:
            for noise_keyword in noise_headers:
                if (noise_keyword.lower() in header_text.lower() or
                    bool(re.match(r'^[^a-zA-Z]*$', header_text)) or
                    bool(re.search(r'\d+†', header_text)) or
                    bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+', header_text))):
                    result["noise"].append(section)
                    categorized = True
                    break
        
        # Additional classification heuristics
        if not categorized:
            word_count = len(header_text.split())
            if word_count <= 3 and '@' not in header_text and '*' not in header_text:
                result["contents"].append(section)
            elif (bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', header_text)) or
                  '@' in header_text or
                  bool(re.match(r'^\d', header_text)) or
                  'correspondence' in header_text.lower()):
                result["noise"].append(section)
            else:
                result["contents"].append(section)
    
    return result


async def main():
    """Main function to process PDF files."""
    input_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/input/002"
    output_dir = "/home/hung/Documents/hung/code/KG_MD/KGChat/data/output/002"
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
        else:
            print(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        # Get list of already processed files
        processed_files = set()
        existing_results = glob.glob(os.path.join(output_dir, "*_result.json"))
        for result_file in existing_results:
            base_name = os.path.basename(result_file).replace("_result.json", "")
            processed_files.add(base_name)
        
        processed_count = 0
        skipped_count = 0
        
        # Create progress bar for PDF processing
        with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
            for pdf_path in pdf_files:
                file_name = Path(pdf_path).stem
                if file_name in processed_files:
                    print(f"Skipping {file_name}.pdf (already processed)")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                    
                pbar.set_description(f"Processing {file_name}.pdf")
                # Step 1: Convert PDF to Markdown
                markdown_path = to_markdown(pdf_path, output_dir)
                # Step 2: Extract headers from Markdown
                head_list = take_head_list(markdown_path)
                # Step 3: Classify headers using GeminiClient
                content_headers, noise_headers = await clasify_headers(head_list, markdown_path)
                # Step 4: Parse Markdown based on classified headers
                result = parse_markdown_headers(markdown_path, content_headers, noise_headers)
                # Step 5: Save the result to a JSON file
                output_json_path = Path(output_dir) / f"{file_name}_result.json"
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"Result saved to: {output_json_path}")
                processed_count += 1
                pbar.update(1)
            
        print(f"\nProcessing complete: {processed_count} files processed, {skipped_count} files skipped.")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())