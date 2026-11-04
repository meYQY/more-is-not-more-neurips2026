"""LLM API client with full request/response logging for reproducibility."""

import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _get_client(config: dict) -> AsyncOpenAI:
    """Get or create an AsyncOpenAI client based on config.

    Clients are cached by (base_url, api_key_env) so different configs
    (e.g. generation vs extraction) can use different providers.
    """
    model_cfg = config["model"]
    base_url = model_cfg["base_url"]
    api_key_env = model_cfg["api_key_env"]
    cache_key = (base_url, api_key_env)

    if cache_key not in _clients:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' is not set. Check your .env file.")
        _clients[cache_key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _clients[cache_key]


async def call_llm_messages(
    messages: list[dict[str, str]],
    config: dict,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call LLM API with arbitrary messages. Returns full request/response/metadata."""
    client = _get_client(config)

    gen = config["generation"]
    params = {
        "model": config["model"]["name"],
        "messages": messages,
        "temperature": gen["temperature"],
        "max_tokens": gen["max_tokens"],
        "top_p": gen["top_p"],
        "frequency_penalty": gen["frequency_penalty"],
        "presence_penalty": gen["presence_penalty"],
    }

    # Provider-specific extra params (e.g. Kimi thinking mode)
    extra_body = config.get("extra_body")
    if extra_body:
        params["extra_body"] = extra_body

    last_error = None
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            completion = await client.chat.completions.create(**params)
            latency_ms = round((time.monotonic() - start) * 1000)
            choice = completion.choices[0]
            return {
                "request": params,
                "response": {
                    "content": choice.message.content,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                    },
                },
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": latency_ms,
                },
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"LLM API failed after {max_retries} retries: {last_error}")


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: dict,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call LLM API with system + user prompt. Convenience wrapper."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return await call_llm_messages(messages, config, max_retries=max_retries)
