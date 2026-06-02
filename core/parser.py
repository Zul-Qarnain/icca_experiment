import re

def extract_label(text, mode='standard'):
    """Extracts the binary digit 0 or 1 from the model's output."""
    if not text:
        return -1
        
    if mode == 'cot':
        # Look specifically inside the <answer> tags
        match = re.search(r'<answer>\s*([01])\s*</answer>', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
    # Standard Mode (or fallback): Look for an isolated 0 or 1
    clean_text = text.strip()
    match = re.search(r'\b([01])\b', clean_text)
    return int(match.group(1)) if match else -1
