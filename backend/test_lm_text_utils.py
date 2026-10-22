"""Causal LM text pipeline and prompt format detection."""

from app.training.lm_text_utils import (
    chunk_text_for_lm,
    detect_prompt_format,
    normalize_causal_lm_text,
    prepare_causal_lm_texts,
)


def test_normalize_collapses_excess_newlines():
    raw = "Title\n\n\n\n\nBody line"
    assert normalize_causal_lm_text(raw) == "Title\n\nBody line"


def test_chunk_wiki_like_article():
    para = "word " * 400
    article = f" = Header = \n\n{para}\n\n{para}"
    chunks = chunk_text_for_lm(article, max_chars=500)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_detect_chat_vs_raw():
    chat = ["User: Hi\nAssistant: Hello there"]
    wiki = [" = Mars = \n\nMars is the fourth planet."]
    assert detect_prompt_format(chat) == "platform_user_assistant"
    assert detect_prompt_format(wiki) == "raw_lm"


def test_prepare_hf_style_expands_rows():
    rows = [" = Wiki = \n\n" + ("sentence. " * 300)]
    out = prepare_causal_lm_texts(rows, source="huggingface")
    assert len(out) > 1
