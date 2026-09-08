#!/usr/bin/env python3
"""OpenAI-SDK compatibility proof.

Uses the *official, unmodified* `openai` Python client against the local
semblance gateway (which forwards to Ollama). If a normal SDK call returns a
correct completion, the gateway is a drop-in OpenAI-compatible endpoint.

    pip install openai
    # gateway running on :8088, backed by Ollama with llama3.2:1b
    python demo/openai_sdk_check.py

Env overrides: SEMBLANCE_URL (default http://127.0.0.1:8088/v1),
SEMBLANCE_MODEL (default llama3.2:1b).
"""

import os
import sys

from openai import OpenAI

base_url = os.environ.get("SEMBLANCE_URL", "http://127.0.0.1:8088/v1")
model = os.environ.get("SEMBLANCE_MODEL", "llama3.2:1b")

# api_key is required by the SDK but ignored by the gateway in open mode.
client = OpenAI(base_url=base_url, api_key=os.environ.get("SEMBLANCE_API_KEY", "not-needed"))

print(f"→ POST {base_url}/chat/completions  (model={model})")
resp = client.chat.completions.create(
    model=model,
    temperature=0,
    messages=[
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ],
)

content = resp.choices[0].message.content
print(f"← assistant: {content!r}")
print(f"  usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}")

# Assertions: a well-formed completion that actually answered.
assert resp.choices, "no choices returned"
assert content and content.strip(), "empty content"
assert "paris" in content.lower(), f"expected 'Paris' in the answer, got: {content!r}"
assert resp.usage.total_tokens > 0, "usage not populated"

print("\nPASS: unmodified OpenAI SDK works against semblance, and the answer is correct.")
sys.exit(0)
