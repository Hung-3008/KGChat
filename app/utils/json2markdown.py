import os 
import json 
import glob 

input_dir = "data/output/001"

glob_pattern = os.path.join(input_dir, "*.json")
json_files = glob.glob(glob_pattern)


def json_to_markdown(json_data):

    markdown_text = ""
    
    if 'contents' in json_data:
        for item in json_data['contents']:
            header = item.get('header', '')
            content = item.get('content', '')
            
            if header:
                markdown_text += f"## {header}\n\n"
            
            if content:
                import re
                content = re.sub(r'<!--.*?-->', '', content)
                
                markdown_text += f"{content}\n\n"
    
    return markdown_text.strip()


for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    markdown_text = json_to_markdown(json_data)
    
    output_file = "data/level_1/" + os.path.basename(json_file).replace('.json', '.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)