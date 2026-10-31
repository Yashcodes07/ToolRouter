## Results (Qwen2.5-7B-Instruct, QLoRA r=32, 1 epoch, 2.2k training examples)

| Split | Model | JSON valid | Tool correct | Args exact | Args core (answerable rows) |
|---|---|---|---|---|---|
| test_seen (n=283) | baseline | 94.7% | 82.7% | 56.2% | 74.7% (n=241) |
| test_seen (n=283) | run1-r32-lr2e-4 | 99.6% | 95.8% | 68.2% | 88.0% (n=241) |
| test_unseen (n=300) | baseline | 91.7% | 82.7% | 58.3% | 75.3% (n=263) |
| test_unseen (n=300) | run1-r32-lr2e-4 | 100.0% | 98.7% | 68.0% | 93.9% (n=263) |

- **Tool correct**: the right tool was chosen. **Args exact**: strict equality on all arguments.
- **Args core**: exact match ignoring free text (file content, observation text, exclude globs),
  arguments equal to the schema default, case, and trailing punctuation.
- **Answerable rows**: rows whose gold values can be read from the request text. Chosen from the
  gold label alone, so both models are scored on the same rows.
- `test_unseen` uses 5 tools that never appear in training.