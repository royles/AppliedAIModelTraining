# Training datasets

Place files here and reference them from **Model Tasks → Dataset** (path relative to this folder).

**Full from-scratch jobs** train a byte-level BPE tokenizer on this text before model training.
Use enough raw text (see the UI guidance for minimum characters vs vocab size).

## Bundled files

| File | Purpose |
|------|---------|
| `starter_corpus.txt` | Default causal-LM corpus (~52k chars). Verbose User/Assistant paragraphs (blank line between turns) about training, Ray, BPE, and this platform—no bracket suffixes. Used by the **scratch_gpt2_tiny** preset and Model Tasks defaults. |
| `example_causal.jsonl` | Small JSONL sample for file-based dataset configuration. |

Regenerate `starter_corpus.txt` after editing seeds:

```bash
python3 scripts/generate_starter_corpus.py
```

## Formats

- **`.txt`**: one training example per line (blank lines are skipped when loading).
- **`.jsonl`**: one JSON object per line; set `text_field` (default `text`) and optional `label_field` for classifiers.
