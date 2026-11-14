import json

system_prompt = (
    "You are Andy, a Network Engineering and Security student at DePaul. "
    "You chat casually about your life, homelab, and AWS projects. "
    "You text like a normal person, use casual language, and never act like an AI assistant or write code."
)

formatted_dataset = []

with open("gemini_training_ready.jsonl", "r") as f:
    for line in f:
        convo = json.loads(line)
        messages = convo.get("messages", [])
        
        # Ensure we have at least a user and assistant turn
        if len(messages) < 2:
            continue
            
        assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        # FILTERING: Drop the row if Gemini wrote code (```) or gave a massive response
        if "```" in assistant_msg or len(assistant_msg.split()) > 100:
            continue
            
        # Rebuild the ChatML array with the strict System Prompt
        chatml_convo = {
            "messages": [
                {"role": "system", "content": system_prompt}
            ]
        }
        
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                chatml_convo["messages"].append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
        formatted_dataset.append(chatml_convo)

# Output the final file for RunPod
with open("qwen_chatml_ready.jsonl", "w") as f:
    for item in formatted_dataset:
        f.write(json.dumps(item) + "\n")

print(f"Kept {len(formatted_dataset)} casual conversations for Qwen fine-tuning.")