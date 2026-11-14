# Gemini Takeout → ChatML fine-tuning pipeline

Turn your own Google Gemini chat history (exported with [Google Takeout](https://takeout.google.com)) into ChatML training data, then fine-tune a small Qwen model on it with LoRA and run it locally in Ollama.

No personal data is included in this repo, only the code.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `gemini_takeout_to_jsonl.py` | Reads a Takeout `.zip`, folder, `MyActivity.json`/`.html`, or Gemini in Workspace conversation files; writes user/assistant pairs (or whole conversations with `--multi-turn`) as JSONL |
| 2 | `prepare_dataset.py` | Redacts emails and AWS access keys, drops turns that depended on images or attachments, and splits long conversations into overlapping 5-turn chunks |
| 3 | `qwentrain.py` | Adds a system prompt and filters out code-heavy or very long replies, producing `qwen_chatml_ready.jsonl` |
| 4 | `train_qwen_unsloth.py` | LoRA fine-tune of Qwen3-4B-Instruct with Unsloth (r=16, 3 epochs), then GGUF Q4_K_M export for Ollama |

```bash
python3 gemini_takeout_to_jsonl.py takeout-XXXX.zip --multi-turn -o gemini_all.jsonl
python3 prepare_dataset.py          # gemini_all.jsonl -> gemini_training_ready.jsonl
python3 qwentrain.py                # -> qwen_chatml_ready.jsonl
python3 train_qwen_unsloth.py       # on a CUDA GPU (tested on an AWS EC2 T4)
```

Optional for huge exports: `pip install ijson tqdm` (streams the file and shows progress).

## Privacy notes

- Redaction is regex-based (emails, `AKIA…` keys). Review the output before training, especially for names, addresses and other secrets.
- Never commit your Takeout export, the generated `.jsonl` files or the trained model; `.gitignore` already blocks them.
- Edit the system prompt in `qwentrain.py` to describe yourself.
