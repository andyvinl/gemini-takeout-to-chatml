import json
import re

INPUT_FILE = "gemini_all.jsonl"
OUTPUT_FILE = "gemini_training_ready.jsonl"
MAX_TURNS = 5 # 5 user-assistant pairs per chunk
OVERLAP = 2   # Number of turns to overlap for context continuity

# Regex patterns for basic PII scrubbing
AWS_KEY_PATTERN = r"(?i)AKIA[0-9A-Z]{16}"
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

# Keywords that indicate the turn relied on a visual attachment
BAD_KEYWORDS = ["screenshot", "attached", "image", "picture", "upload"]

def clean_text(text):
    text = re.sub(AWS_KEY_PATTERN, "[REDACTED_AWS_KEY]", text)
    text = re.sub(EMAIL_PATTERN, "[REDACTED_EMAIL]", text)
    return text

def contains_bad_keywords(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in BAD_KEYWORDS)

with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        chat = json.loads(line)
        messages = chat.get("messages", [])
        
        # Group messages into user-assistant pairs (1 turn = 1 pair)
        pairs = []
        for i in range(0, len(messages)-1, 2):
            if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                pairs.append((messages[i], messages[i+1]))
        
        # Filter and clean the text within each pair
        valid_pairs = []
        for user_msg, asst_msg in pairs:
            u_text = clean_text(user_msg['content'])
            a_text = clean_text(asst_msg['content'])
            
            if not contains_bad_keywords(u_text) and not contains_bad_keywords(a_text):
                valid_pairs.append([
                    {"role": "user", "content": u_text},
                    {"role": "assistant", "content": a_text}
                ])
        
        if not valid_pairs:
            continue
            
        # Chunking: Sliding window over the valid pairs
        stride = MAX_TURNS - OVERLAP
        if stride < 1: stride = 1
        
        for i in range(0, len(valid_pairs), stride):
            chunk_pairs = valid_pairs[i:i + MAX_TURNS]
            
            # Flatten the pairs back into a standard message array
            chunk_messages = []
            for p in chunk_pairs:
                chunk_messages.extend(p)
            
            # Write the chunk as a new JSONL line
            outfile.write(json.dumps({"messages": chunk_messages}) + "\n")
            
            # If we've hit the end of the pairs, stop chunking this chat
            if i + MAX_TURNS >= len(valid_pairs):
                break
                
print(f"Processing complete. Data saved to {OUTPUT_FILE}")
