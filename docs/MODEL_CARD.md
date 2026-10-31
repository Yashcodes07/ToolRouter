---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: apache-2.0
language:
- en
tags:
- tool-calling
- function-calling
- mcp
- lora
- qlora
- unsloth
datasets:
- Yashcodes07/toolrouter-mcp-calls-v2
pipeline_tag: text-generation
---

# ToolRouter: Qwen2.5-7B-Instruct LoRA for MCP tool calling

A QLoRA adapter that teaches Qwen2.5-7B-Instruct to turn a plain-English request into a single
[MCP](https://modelcontextprotocol.io) tool call: pick the right tool from a list of schemas and fill in its
arguments as valid JSON. It is the small model in a cascade: it handles most requests cheaply, and a larger
model takes over when its output fails schema validation.

## Results

Evaluation uses the same prompt (Qwen's native tool-calling chat template with **all 23 tool schemas** in the
prompt) and the same scorer for every model. `test_unseen` contains 5 tools that never appear in training.

| Split | Model | JSON valid | Tool correct | Args exact | Args core (answerable rows) |
|---|---|---|---|---|---|
| test_seen (n=283) | Base Qwen2.5-7B-Instruct | 94.7% | 82.7% | 56.2% | 74.7% (n=241) |
| | Adapter v1 (r=32) | 99.6% | 95.8% | 68.2% | 88.0% (n=241) |
| | Adapter v1 (r=64) | 99.3% | 94.0% | 67.8% | 86.7% (n=241) |
| | **Adapter v2 (r=32)** | TBD | TBD | TBD | TBD |
| test_unseen (n=300) | Base Qwen2.5-7B-Instruct | 91.7% | 82.7% | 58.3% | 75.3% (n=263) |
| | Adapter v1 (r=32) | 100.0% | 98.7% | 68.0% | 93.9% (n=263) |
| | Adapter v1 (r=64) | 100.0% | 99.3% | 68.7% | 94.7% (n=263) |
| | **Adapter v2 (r=32)** | TBD | TBD | TBD | TBD |

Metric definitions:

- **JSON valid**: the output contains a parseable tool call with a name and an arguments object.
- **Tool correct**: the called tool is the right one.
- **Args exact**: the right tool and every argument identical to the reference.
- **Args core**: right tool, and arguments identical after dropping free text (file content, observation text,
  exclude globs), dropping arguments equal to the schema default, and ignoring case and trailing punctuation.
  Reported on **answerable rows**, meaning rows whose reference values (full paths, names, titles, relation
  types, task ids) can be read from the request text. Answerability is decided from the reference alone, so
  every model is scored on the same rows.

Rank study: going from r=32 to r=64 gave no measurable gain (args core 88.0% vs 86.7% seen, 93.9% vs 94.7%
unseen, paired sign test p > 0.5) and slightly worse tool selection on seen tools (6 rows vs 0, p = 0.03), so
r=32 was used for the final model.

## Training data

Synthetic requests paired with tool calls for 23 tools on three MCP servers: the official filesystem server,
the official memory (knowledge graph) server, and a small SQLite task server. Requests were generated with
`gpt-oss-120b` and filtered hard: JSON-schema valid, argument values grounded in the request text,
deduplicated, and **executed against the real servers**. Dataset: `Yashcodes07/toolrouter-mcp-calls-v2`
(train 2,643, val 243, test_seen 283, test_unseen 300). Five tools (`move_file`, `search_files`, `delete_task`,
`add_observations`, `delete_relations`) are held out of training entirely.

Version 2 of the data removed training rows whose reference values could not be read from the request, and
regenerated rows for the weakest tools with requests that state those values.

## Training

- Method: QLoRA with Unsloth and TRL `SFTTrainer`; base weights in 4-bit NF4.
- LoRA: r=32, alpha=64, dropout 0, all attention and MLP projections (`q,k,v,o,gate,up,down`).
- 1 epoch, effective batch size 16, learning rate 2e-4 (cosine, 5% warmup), weight decay 0.01, `adamw_8bit`,
  max sequence length 3072, fp16, seed 3407.
- Loss on the assistant reply only. Each training prompt contains the correct tool plus 5 to 9 other tools
  from the 18 seen tools, shuffled; held-out tools never appear in a training prompt.
- Hardware: one free Colab T4, about 2.8 hours.

## How to use

The adapter expects Qwen's tool-calling chat template and answers with
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
model = PeftModel.from_pretrained(model, "Yashcodes07/toolrouter-qwen2.5-7b-lora-v2-r32-lr2e-4")

tools = [{"type": "function", "function": {
    "name": "create_directory", "description": "Create a directory.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
prompt = tok.apply_chat_template([{"role": "user", "content": "Make a folder called notes/ideas"}],
                                 tools=tools, add_generation_prompt=True, tokenize=False)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=256, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Limitations

- Synthetic data from one generator model; real users phrase requests differently.
- Only 23 tools on 3 servers. The held-out tools resemble the seen ones (same servers and schema style), so
  `test_unseen` shows generalization to new tools of a familiar kind, not to arbitrary APIs.
- Single-turn calls only: no multi-step plans, no parallel calls, no clarifying questions.
- Strict exact-match accuracy is about 68%. Many remaining misses are valid alternatives (path naming, glob
  patterns, keyword choice, relation names) or free text; a few are real mistakes such as confusing
  `list_directory` with `directory_tree` or reversing a relation. Do not use it to run destructive tools
  (`delete_*`, `write_file`) without validation or confirmation.
- Evaluated with all 23 tools in the prompt; behaviour with other tool sets is untested.