import re

def update_pdf():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    start_pattern = r'            def generate_pdf_report\(sd, df, insights, summary, now_str,'
    end_pattern = r'                return buf\.read\(\)\n'
    
    start_match = re.search(start_pattern, content)
    end_match = re.search(end_pattern, content)
    
    start_pos = start_match.start()
    end_pos = end_match.end()

    old_func = content[start_pos:end_pos]
    
    with open('patch.txt', 'r', encoding='utf-8') as pf:
        patch_text = pf.read()

    start_patch_match = re.search(start_pattern, patch_text)
    end_patch_match = re.search(end_pattern, patch_text)

    new_func = patch_text[start_patch_match.start():end_patch_match.end()]

    new_content = content[:start_pos] + new_func + content[end_pos:]

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Updated app.py!")

update_pdf()
