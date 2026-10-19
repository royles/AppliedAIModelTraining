# Training datasets

Place files here and reference them from **Model Tasks → Dataset** (path relative to this folder).

**Full from-scratch jobs** train a byte-level BPE tokenizer on this text before model training.
Use enough raw text (see the UI guidance for minimum characters vs vocab size); the built-in
demo corpus is not sufficient for tokenizer training.

- **`.txt`**: one training example per line
- **`.jsonl`**: one JSON object per line; set `text_field` (default `text`) and optional `label_field` for classifiers

Example: `example_causal.jsonl`
