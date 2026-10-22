"""Inference prompts match training format."""

from app.inference.prompting import (
    platform_messages_to_prompt,
    raw_lm_messages_to_prompt,
    use_platform_prompt_format,
)


def test_raw_lm_prompt_uses_user_text_only():
    prompt = raw_lm_messages_to_prompt([{"role": "user", "content": "hello"}])
    assert prompt == "hello"
    assert "User:" not in prompt


def test_platform_chat_prompt():
    prompt = platform_messages_to_prompt([{"role": "user", "content": "Test"}])
    assert "User: Test" in prompt
    assert prompt.strip().endswith("Assistant:")


def test_use_platform_format_only_when_configured():
    assert use_platform_prompt_format("", {"prompt_format": "platform_user_assistant"})
    assert not use_platform_prompt_format("", {"init_mode": "from_scratch", "task_type": "causal_lm"})
    assert use_platform_prompt_format("", {"prompt_format": "raw_lm"}) is False
